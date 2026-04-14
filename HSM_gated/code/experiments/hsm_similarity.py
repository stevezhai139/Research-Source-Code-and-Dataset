"""
hsm_similarity.py
=================
HSM (Hierarchical Similarity Measurement) — Core Module
HSM: Multi-Scale Workload Similarity for Database Index Management

Implements 5-dimensional similarity:
  S_R : Relation (table) similarity        — Jaccard over tables accessed
  S_V : Value-range similarity             — overlap of numeric/date predicates
  S_T : Template (query structure) similarity — Jaccard over query templates
  S_A : Attribute (column) similarity      — Jaccard over columns referenced
  S_P : Predicate similarity               — Jaccard over predicate operators

Combined:
  HSM(W_a, W_b) = w_R*S_R + w_V*S_V + w_T*S_T + w_A*S_A + w_P*S_P

Decision rule (T4 in HSM):
  If HSM(W_prev, W_curr) < theta  →  trigger index advisor
  Else                            →  reuse existing indexes

References:
  HSM, Section 3 (HSM Framework), Theorems T1–T9
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ─── Default weights (equal, as in HSM baseline) ─────────────────────────
DEFAULT_WEIGHTS = {
    'w_R': 0.20,
    'w_V': 0.20,
    'w_T': 0.20,
    'w_A': 0.20,
    'w_P': 0.20,
}

# ─── Decision threshold θ (HSM, Section 3.4) ─────────────────────────────
DEFAULT_THETA = 0.75


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class QueryFeatures:
    """Extracted features from a single SQL query."""
    raw_sql:    str
    tables:     set  = field(default_factory=set)   # S_R
    columns:    set  = field(default_factory=set)   # S_A
    predicates: set  = field(default_factory=set)   # S_P (operators)
    ranges:     list = field(default_factory=list)  # S_V [(col, lo, hi)]
    template:   str  = ""                           # S_T (normalized SQL)


@dataclass
class WorkloadWindow:
    """A sliding window of queries representing a workload phase."""
    queries:  List[QueryFeatures] = field(default_factory=list)
    window_id: int = 0

    # Aggregated features
    @property
    def tables(self) -> set:
        return set().union(*[q.tables for q in self.queries]) if self.queries else set()

    @property
    def columns(self) -> set:
        return set().union(*[q.columns for q in self.queries]) if self.queries else set()

    @property
    def predicates(self) -> set:
        return set().union(*[q.predicates for q in self.queries]) if self.queries else set()

    @property
    def templates(self) -> set:
        return {q.template for q in self.queries}

    @property
    def ranges(self) -> list:
        r = []
        for q in self.queries:
            r.extend(q.ranges)
        return r


# ─── Feature Extraction ───────────────────────────────────────────────────────

# TPC-H table names
TPCH_TABLES = {
    'lineitem', 'orders', 'customer', 'part', 'partsupp',
    'supplier', 'nation', 'region'
}

PREDICATE_OPS = {'=', '<', '>', '<=', '>=', '<>', '!=',
                 'between', 'like', 'in', 'not in', 'is null', 'is not null'}


def extract_features(sql: str) -> QueryFeatures:
    """Extract HSM-relevant features from a SQL string."""
    sql_lower = sql.lower()
    qf = QueryFeatures(raw_sql=sql)

    # S_R: Tables
    for tbl in TPCH_TABLES:
        if re.search(r'\b' + tbl + r'\b', sql_lower):
            qf.tables.add(tbl)

    # S_A: Columns — match patterns like "alias.column" or bare column names
    col_matches = re.findall(
        r'\b([a-z_]+\.[a-z_]+)\b|'
        r'\b(l_\w+|o_\w+|c_\w+|p_\w+|ps_\w+|s_\w+|n_\w+|r_\w+)\b',
        sql_lower
    )
    for m in col_matches:
        col = m[0] if m[0] else m[1]
        if col:
            qf.columns.add(col)

    # S_P: Predicate operators
    for op in PREDICATE_OPS:
        if op in sql_lower:
            qf.predicates.add(op)

    # S_V: Numeric and date ranges
    # BETWEEN patterns: col BETWEEN val AND val
    between_matches = re.findall(
        r'(\w+)\s+between\s+([\d\'\-\.]+)\s+and\s+([\d\'\-\.]+)',
        sql_lower
    )
    for col, lo, hi in between_matches:
        qf.ranges.append((col, lo, hi))

    # Comparison predicates: col >= val
    comp_matches = re.findall(
        r'(\w+)\s*[<>]=?\s*([\d\'\-\.]+)',
        sql_lower
    )
    for col, val in comp_matches:
        qf.ranges.append((col, val, val))

    # S_T: Query template (normalize literals to placeholders)
    template = re.sub(r"'[^']*'",     '?', sql_lower)  # string literals
    template = re.sub(r'\b\d+\.?\d*\b', '?', template)  # numeric literals
    template = re.sub(r'\s+', ' ', template).strip()
    qf.template = template

    return qf


# ─── Similarity Dimensions ────────────────────────────────────────────────────

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|. Returns 1.0 if both empty."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def s_r(w_a: WorkloadWindow, w_b: WorkloadWindow) -> float:
    """S_R: Relation (table) similarity."""
    return jaccard(w_a.tables, w_b.tables)


def s_a(w_a: WorkloadWindow, w_b: WorkloadWindow) -> float:
    """S_A: Attribute (column) similarity."""
    return jaccard(w_a.columns, w_b.columns)


def s_p(w_a: WorkloadWindow, w_b: WorkloadWindow) -> float:
    """S_P: Predicate operator similarity."""
    return jaccard(w_a.predicates, w_b.predicates)


def s_t(w_a: WorkloadWindow, w_b: WorkloadWindow) -> float:
    """S_T: Query template similarity."""
    return jaccard(w_a.templates, w_b.templates)


def s_v(w_a: WorkloadWindow, w_b: WorkloadWindow) -> float:
    """
    S_V: Value-range similarity.
    Measures overlap of numeric/date predicate ranges across windows.
    Uses average pairwise range overlap for matching columns.
    """
    ranges_a = w_a.ranges
    ranges_b = w_b.ranges

    if not ranges_a and not ranges_b:
        return 1.0
    if not ranges_a or not ranges_b:
        return 0.0

    # Group ranges by column
    def group_by_col(ranges):
        d = {}
        for col, lo, hi in ranges:
            d.setdefault(col, []).append((lo, hi))
        return d

    ga = group_by_col(ranges_a)
    gb = group_by_col(ranges_b)

    common_cols = set(ga.keys()) & set(gb.keys())
    if not common_cols:
        return 0.0

    total_sim = 0.0
    for col in common_cols:
        # Compute average overlap for this column's range pairs
        col_sims = []
        for lo_a, hi_a in ga[col]:
            for lo_b, hi_b in gb[col]:
                sim = _range_overlap(lo_a, hi_a, lo_b, hi_b)
                col_sims.append(sim)
        total_sim += (sum(col_sims) / len(col_sims) if col_sims else 0.0)

    all_cols = set(ga.keys()) | set(gb.keys())
    return total_sim / len(all_cols) if all_cols else 1.0


def _try_float(v: str) -> Optional[float]:
    """Try converting a string value to float (handles dates roughly)."""
    try:
        return float(v)
    except (ValueError, TypeError):
        # Try date as days since epoch (rough)
        try:
            from datetime import datetime
            return datetime.strptime(v.strip("'"), '%Y-%m-%d').timestamp()
        except Exception:
            return None


def _range_overlap(lo_a, hi_a, lo_b, hi_b) -> float:
    """
    Compute overlap ratio between two 1-D ranges.
    Returns 1.0 if identical, 0.0 if no overlap.
    """
    a_lo = _try_float(str(lo_a))
    a_hi = _try_float(str(hi_a))
    b_lo = _try_float(str(lo_b))
    b_hi = _try_float(str(hi_b))

    if None in (a_lo, a_hi, b_lo, b_hi):
        return 1.0 if str(lo_a) == str(lo_b) else 0.0

    if a_lo == a_hi and b_lo == b_hi:
        return 1.0 if a_lo == b_lo else 0.0

    span_a = abs(a_hi - a_lo) if a_hi != a_lo else 1.0
    span_b = abs(b_hi - b_lo) if b_hi != b_lo else 1.0

    overlap_lo = max(a_lo, b_lo)
    overlap_hi = min(a_hi, b_hi)
    overlap    = max(0.0, overlap_hi - overlap_lo)
    union_span = max(a_hi, b_hi) - min(a_lo, b_lo)

    return overlap / union_span if union_span > 0 else 1.0


# ─── Combined HSM Score ───────────────────────────────────────────────────────

def hsm_score(
    w_a: WorkloadWindow,
    w_b: WorkloadWindow,
    weights: Dict[str, float] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute HSM similarity between two workload windows.

    Returns
    -------
    (score, components)
        score      : float in [0, 1]
        components : dict with individual dimension scores
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    dims = {
        'S_R': s_r(w_a, w_b),
        'S_V': s_v(w_a, w_b),
        'S_T': s_t(w_a, w_b),
        'S_A': s_a(w_a, w_b),
        'S_P': s_p(w_a, w_b),
    }

    score = (
        weights['w_R'] * dims['S_R'] +
        weights['w_V'] * dims['S_V'] +
        weights['w_T'] * dims['S_T'] +
        weights['w_A'] * dims['S_A'] +
        weights['w_P'] * dims['S_P']
    )

    return score, dims


def should_trigger_advisor(
    w_prev: WorkloadWindow,
    w_curr: WorkloadWindow,
    theta: float = DEFAULT_THETA,
    weights: Dict[str, float] = None
) -> Tuple[bool, float, Dict[str, float]]:
    """
    HSM gating decision (HSM, T4).

    Returns (trigger, score, components):
      trigger = True  → workload drift detected, run index advisor
      trigger = False → workload stable, skip advisor
    """
    if w_prev is None or not w_prev.queries:
        return True, 0.0, {}  # No history — always trigger on first window

    score, dims = hsm_score(w_prev, w_curr, weights)
    trigger = score < theta
    return trigger, score, dims


# ─── Utility ──────────────────────────────────────────────────────────────────

def build_window(sql_list: List[str], window_id: int = 0) -> WorkloadWindow:
    """Build a WorkloadWindow from a list of SQL strings."""
    features = [extract_features(sql) for sql in sql_list]
    return WorkloadWindow(queries=features, window_id=window_id)


# ─── Quick Self-Test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    q_lineitem = """
        SELECT l_returnflag, l_linestatus, SUM(l_quantity)
        FROM lineitem
        WHERE l_shipdate <= '1998-09-01'
        GROUP BY l_returnflag, l_linestatus
    """
    q_orders = """
        SELECT o_orderpriority, COUNT(*) AS order_count
        FROM orders
        WHERE o_orderdate BETWEEN '1993-07-01' AND '1993-10-01'
        GROUP BY o_orderpriority
    """
    q_customer = """
        SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount))
        FROM customer, orders, lineitem
        WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
          AND c_nationkey = 15
        GROUP BY c_custkey, c_name
    """

    w1 = build_window([q_lineitem, q_lineitem], window_id=1)
    w2 = build_window([q_orders, q_customer], window_id=2)

    score, dims = hsm_score(w1, w2)
    trigger, _, _ = should_trigger_advisor(w1, w2)

    print(f"HSM Score:  {score:.4f}")
    print(f"Components: {dims}")
    print(f"Trigger advisor (θ=0.75): {trigger}")
