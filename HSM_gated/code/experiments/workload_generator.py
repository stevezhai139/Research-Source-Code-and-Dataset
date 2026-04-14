"""
workload_generator.py
=====================
TPC-H Workload Sequence Generator with Deliberate Phase Shifts
HSM Throughput Experiments  (v2 — without-replacement + parameter pools)

DESIGN CHANGES FROM v1:
────────────────────────

  Fix 1 — Query repetition within windows (Birthday Problem):
    v1: rng.choice(available) with replacement.
        With window_size=5 and |phase|=5 query types:
          P(all-distinct in window) = 5!/5⁵ ≈ 3.8%
          P(≥1 duplicate in window) ≈ 96.2%
        Phase C (3 types, window 5) always has duplicates by Pigeonhole.
        Repeated queries within the same window access the same data pages
        → second execution hits page cache, biasing all conditions equally
        but making advisor benefit harder to observe (baseline warms up fast).
    v2: Shuffled round-robin (_gen_phase) — without replacement.
        Each window of 5 contains each Phase A/B query type exactly once.
        P(intra-window type duplication) = 0% for Phase A and B.
        Phase C (3 types < window_size) still has unavoidable repeats, minimised.

  Fix 2 — Fixed SQL parameters → inter-block cache reuse:
    v1: All SQL used fixed literals (l_shipdate <= '1998-09-02',
        c_mktsegment = 'BUILDING', etc.).  Every block ran the exact same
        SQL strings → data pages loaded by block 1 remained warm in
        shared_buffers at block 10 → baseline gained systematic cache warmth
        advantage across blocks; advisor conditions were periodically
        disrupted by CREATE INDEX page evictions.
    v2: Parameter pools (3 validated variants per active query).
        get_workload_trace(seed=block) draws one variant per execution.
        Callers pass seed=block → each block uses a different parameter set
        → inter-block page-cache reuse is eliminated.

Structure:
  Phase A  (LINEITEM/ORDERS heavy) : Q1, Q3, Q4, Q6, Q12
  Phase B  (CUSTOMER/PART heavy)   : Q10, Q13, Q14, Q17, Q18
  Phase A_repeat                   : same as Phase A (tests HSM stability)
  Phase C  (Mixed analytical)      : Q2, Q8, Q11

Usage:
  from workload_generator import get_workload_trace, PHASE_A, PHASE_B, PHASE_C

  # Caller passes seed=block for inter-block parameter diversity:
  trace = get_workload_trace(queries_per_phase=30, seed=block)
"""

import random
from typing import List, Tuple

# ─── Parameter Pools ───────────────────────────────────────────────────────────
# 3 validated variants per active query.
#
# Selection criteria:
#   • Same tables and columns accessed in every variant
#     → HSM feature extraction (S_R, S_V, S_P) is unaffected by parameter choice
#   • All variants produce non-trivial result sets at SF=0.2, 1.0, 3.0, 10.0
#   • Variants span different data ranges → different OS/buffer-cache pages
#
# Pool size = 3:
#   P(same parameter in two consecutive blocks) = 1/3 = 33%
#   This is sufficient to break the systematic inter-block cache reuse
#   that occurs with pool size = 1 (100% reuse).
#   Increasing to 5+ variants yields diminishing returns at SF=10 where
#   the 60M-row lineitem table never fully fits in the 8GB Docker container.

QUERY_PARAM_POOLS = {
    # ── Phase A ────────────────────────────────────────────────────────────────

    # Q1  Pricing Summary — vary shipdate cutoff (days before 1998-12-01)
    "Q1":  [
        {"delta": "90"},
        {"delta": "108"},
        {"delta": "120"},
    ],

    # Q3  Shipping Priority — vary market segment + split date
    "Q3":  [
        {"segment": "BUILDING",   "cutdate": "1995-03-15"},
        {"segment": "AUTOMOBILE", "cutdate": "1995-06-15"},
        {"segment": "MACHINERY",  "cutdate": "1995-09-15"},
    ],

    # Q4  Order Priority Checking — vary quarter start
    "Q4":  [
        {"qstart": "1993-07-01"},
        {"qstart": "1993-10-01"},
        {"qstart": "1994-01-01"},
    ],

    # Q6  Forecast Revenue Change — vary year
    "Q6":  [
        {"year": "1994"},
        {"year": "1995"},
        {"year": "1996"},
    ],

    # Q12 Shipping Modes — vary receiptdate year
    "Q12": [
        {"year": "1994"},
        {"year": "1995"},
        {"year": "1996"},
    ],

    # ── Phase B ────────────────────────────────────────────────────────────────

    # Q10 Returned Part Revenue — vary quarter
    "Q10": [
        {"qstart": "1993-10-01"},
        {"qstart": "1994-01-01"},
        {"qstart": "1994-04-01"},
    ],

    # Q13 Customer Distribution — vary order-comment exclusion pattern
    #     All three patterns appear in TPC-H generated data
    "Q13": [
        {"pattern": "%special%requests%"},
        {"pattern": "%unusual%packages%"},
        {"pattern": "%express%requests%"},
    ],

    # Q14 Promotion Effect — vary shipdate month
    "Q14": [
        {"mstart": "1995-09-01"},
        {"mstart": "1995-10-01"},
        {"mstart": "1995-11-01"},
    ],

    # Q17 Small Quantity Order Revenue — vary brand + container type
    "Q17": [
        {"brand": "Brand#23", "container": "MED BOX"},
        {"brand": "Brand#12", "container": "SM CASE"},
        {"brand": "Brand#34", "container": "LG BOX"},
    ],

    # Q18 Large Volume Customer — vary quantity threshold
    "Q18": [
        {"threshold": "300"},
        {"threshold": "312"},
        {"threshold": "315"},
    ],

    # ── Phase C ────────────────────────────────────────────────────────────────

    # Q2  Minimum Cost Supplier — vary part size
    "Q2":  [
        {"size": "15"},
        {"size": "23"},
        {"size": "36"},
    ],

    # Q8  National Market Share — vary part type
    "Q8":  [
        {"ptype": "ECONOMY ANODIZED STEEL"},
        {"ptype": "PROMO ANODIZED STEEL"},
        {"ptype": "ECONOMY BURNISHED COPPER"},
    ],

    # Q11 Important Stock Identification — vary supplying nation
    #     {nation} appears in BOTH the outer WHERE and the inner subquery
    #     HAVING; both are replaced with the same value by str.format()
    "Q11": [
        {"nation": "GERMANY"},
        {"nation": "FRANCE"},
        {"nation": "JAPAN"},
    ],

    # ── Inactive queries (defined but not assigned to any phase) ───────────────
    # Pool [{}] → str.format(**{}) is a no-op on a template with no placeholders
    "Q5":  [{}], "Q7": [{}], "Q9": [{}], "Q20": [{}], "Q21": [{}],
}


# ─── SQL Templates ────────────────────────────────────────────────────────────
# Curly-brace placeholders map to keys in QUERY_PARAM_POOLS.
# All other SQL syntax is unchanged from v1.
# Column references are identical to v1 → HSM feature extraction is unaffected.

TPCH_TEMPLATES = {

    # ── Phase A: LINEITEM / ORDERS heavy ──────────────────────────────────────

    "Q1": """
        SELECT
            l_returnflag, l_linestatus,
            SUM(l_quantity)            AS sum_qty,
            SUM(l_extendedprice)       AS sum_base_price,
            SUM(l_extendedprice*(1-l_discount))           AS sum_disc_price,
            SUM(l_extendedprice*(1-l_discount)*(1+l_tax)) AS sum_charge,
            AVG(l_quantity)            AS avg_qty,
            AVG(l_extendedprice)       AS avg_price,
            AVG(l_discount)            AS avg_disc,
            COUNT(*)                   AS count_order
        FROM lineitem
        WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '{delta} day'
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus
    """,

    "Q3": """
        SELECT
            l_orderkey,
            SUM(l_extendedprice*(1-l_discount)) AS revenue,
            o_orderdate, o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = '{segment}'
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < DATE '{cutdate}'
          AND l_shipdate  > DATE '{cutdate}'
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10
    """,

    "Q4": """
        SELECT o_orderpriority, COUNT(*) AS order_count
        FROM orders
        WHERE o_orderdate >= DATE '{qstart}'
          AND o_orderdate <  DATE '{qstart}' + INTERVAL '3 month'
          AND EXISTS (
              SELECT * FROM lineitem
              WHERE l_orderkey = o_orderkey
                AND l_commitdate < l_receiptdate
          )
        GROUP BY o_orderpriority
        ORDER BY o_orderpriority
    """,

    "Q6": """
        SELECT SUM(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '{year}-01-01'
          AND l_shipdate <  DATE '{year}-01-01' + INTERVAL '1 year'
          AND l_discount BETWEEN 0.05 AND 0.07
          AND l_quantity < 24
    """,

    "Q12": """
        SELECT l_shipmode,
               SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
               SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
        FROM orders, lineitem
        WHERE o_orderkey = l_orderkey
          AND l_shipmode IN ('MAIL', 'SHIP')
          AND l_commitdate < l_receiptdate
          AND l_shipdate   < l_commitdate
          AND l_receiptdate >= DATE '{year}-01-01'
          AND l_receiptdate <  DATE '{year}-01-01' + INTERVAL '1 year'
        GROUP BY l_shipmode
        ORDER BY l_shipmode
    """,

    # ── Phase B: CUSTOMER / PART heavy ────────────────────────────────────────

    "Q10": """
        SELECT c_custkey, c_name,
               SUM(l_extendedprice*(1-l_discount)) AS revenue,
               c_acctbal, n_name, c_address, c_phone, c_comment
        FROM customer, orders, lineitem, nation
        WHERE c_custkey  = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate >= DATE '{qstart}'
          AND o_orderdate <  DATE '{qstart}' + INTERVAL '3 month'
          AND l_returnflag = 'R'
          AND c_nationkey = n_nationkey
        GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
        ORDER BY revenue DESC
        LIMIT 20
    """,

    "Q13": """
        SELECT c_count, COUNT(*) AS custdist
        FROM (
            SELECT c_custkey, COUNT(o_orderkey) AS c_count
            FROM customer LEFT OUTER JOIN orders
                ON c_custkey = o_custkey AND o_comment NOT LIKE '{pattern}'
            GROUP BY c_custkey
        ) AS c_orders
        GROUP BY c_count
        ORDER BY custdist DESC, c_count DESC
    """,

    "Q14": """
        SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%'
                                 THEN l_extendedprice*(1-l_discount)
                                 ELSE 0 END)
               / SUM(l_extendedprice*(1-l_discount)) AS promo_revenue
        FROM lineitem, part
        WHERE l_partkey = p_partkey
          AND l_shipdate >= DATE '{mstart}'
          AND l_shipdate <  DATE '{mstart}' + INTERVAL '1 month'
    """,

    "Q17": """
        SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly
        FROM lineitem, part
        WHERE p_partkey = l_partkey
          AND p_brand     = '{brand}'
          AND p_container = '{container}'
          AND l_quantity < (
              SELECT 0.2 * AVG(l_quantity)
              FROM lineitem
              WHERE l_partkey = p_partkey
          )
    """,

    "Q18": """
        SELECT c_name, c_custkey, o_orderkey, o_orderdate,
               o_totalprice, SUM(l_quantity)
        FROM customer, orders, lineitem
        WHERE o_orderkey IN (
            SELECT l_orderkey FROM lineitem
            GROUP BY l_orderkey HAVING SUM(l_quantity) > {threshold}
        )
          AND c_custkey  = o_custkey
          AND o_orderkey = l_orderkey
        GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
        ORDER BY o_totalprice DESC, o_orderdate
        LIMIT 100
    """,

    # ── Phase C: Mixed analytical ──────────────────────────────────────────────

    "Q2": """
        SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr,
               s_address, s_phone, s_comment
        FROM part, supplier, partsupp, nation, region
        WHERE p_partkey   = ps_partkey
          AND s_suppkey   = ps_suppkey
          AND p_size       = {size}
          AND p_type LIKE '%BRASS'
          AND s_nationkey = n_nationkey
          AND n_regionkey = r_regionkey
          AND r_name      = 'EUROPE'
          AND ps_supplycost = (
              SELECT MIN(ps_supplycost)
              FROM partsupp, supplier, nation, region
              WHERE p_partkey   = ps_partkey
                AND s_suppkey   = ps_suppkey
                AND s_nationkey = n_nationkey
                AND n_regionkey = r_regionkey
                AND r_name      = 'EUROPE'
          )
        ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
        LIMIT 100
    """,

    "Q8": """
        SELECT o_year,
               SUM(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END)
               / SUM(volume) AS mkt_share
        FROM (
            SELECT EXTRACT(YEAR FROM o_orderdate) AS o_year,
                   l_extendedprice*(1-l_discount) AS volume,
                   n2.n_name AS nation
            FROM part, supplier, lineitem, orders, customer,
                 nation n1, nation n2, region
            WHERE p_partkey   = l_partkey
              AND s_suppkey   = l_suppkey
              AND l_orderkey  = o_orderkey
              AND o_custkey   = c_custkey
              AND c_nationkey = n1.n_nationkey
              AND n1.n_regionkey = r_regionkey
              AND r_name      = 'AMERICA'
              AND s_nationkey = n2.n_nationkey
              AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
              AND p_type      = '{ptype}'
        ) AS all_nations
        GROUP BY o_year
        ORDER BY o_year
    """,

    "Q11": """
        SELECT ps_partkey,
               SUM(ps_supplycost * ps_availqty) AS value
        FROM partsupp, supplier, nation
        WHERE ps_suppkey  = s_suppkey
          AND s_nationkey = n_nationkey
          AND n_name      = '{nation}'
        GROUP BY ps_partkey
        HAVING SUM(ps_supplycost * ps_availqty) > (
            SELECT SUM(ps_supplycost * ps_availqty) * 0.0001
            FROM partsupp, supplier, nation
            WHERE ps_suppkey  = s_suppkey
              AND s_nationkey = n_nationkey
              AND n_name      = '{nation}'
        )
        ORDER BY value DESC
    """,

    # ── Inactive queries (defined for completeness; not assigned to any phase) ─

    "Q5": """
        SELECT n_name, SUM(l_extendedprice*(1-l_discount)) AS revenue
        FROM customer, orders, lineitem, supplier, nation, region
        WHERE c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND l_suppkey  = s_suppkey
          AND c_nationkey = s_nationkey
          AND s_nationkey = n_nationkey
          AND n_regionkey = r_regionkey
          AND r_name = 'ASIA'
          AND o_orderdate >= DATE '1994-01-01'
          AND o_orderdate <  DATE '1994-01-01' + INTERVAL '1 year'
        GROUP BY n_name
        ORDER BY revenue DESC
    """,

    "Q7": """
        SELECT supp_nation, cust_nation, l_year,
               SUM(volume) AS revenue
        FROM (
            SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation,
                   EXTRACT(YEAR FROM l_shipdate) AS l_year,
                   l_extendedprice * (1 - l_discount) AS volume
            FROM supplier, lineitem, orders, customer,
                 nation n1, nation n2
            WHERE s_suppkey = l_suppkey
              AND o_orderkey = l_orderkey
              AND c_custkey = o_custkey
              AND s_nationkey = n1.n_nationkey
              AND c_nationkey = n2.n_nationkey
              AND ((n1.n_name = 'FRANCE' AND n2.n_name = 'GERMANY')
                OR (n1.n_name = 'GERMANY' AND n2.n_name = 'FRANCE'))
              AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
        ) AS shipping
        GROUP BY supp_nation, cust_nation, l_year
        ORDER BY supp_nation, cust_nation, l_year
    """,

    "Q9": """
        SELECT nation, o_year, SUM(amount) AS sum_profit
        FROM (
            SELECT n_name AS nation,
                   EXTRACT(YEAR FROM o_orderdate) AS o_year,
                   l_extendedprice*(1-l_discount) - ps_supplycost*l_quantity AS amount
            FROM part, supplier, lineitem, partsupp, orders, nation
            WHERE s_suppkey  = l_suppkey
              AND ps_suppkey = l_suppkey
              AND ps_partkey = l_partkey
              AND p_partkey  = l_partkey
              AND o_orderkey = l_orderkey
              AND s_nationkey = n_nationkey
              AND p_name LIKE '%green%'
        ) AS profit
        GROUP BY nation, o_year
        ORDER BY nation, o_year DESC
    """,

    "Q20": """
        SELECT s_name, s_address
        FROM supplier, nation
        WHERE s_suppkey IN (
            SELECT ps_suppkey FROM partsupp
            WHERE ps_partkey IN (
                SELECT p_partkey FROM part WHERE p_name LIKE 'forest%'
            )
              AND ps_availqty > (
                  SELECT 0.5 * SUM(l_quantity)
                  FROM lineitem
                  WHERE l_partkey = ps_partkey
                    AND l_suppkey = ps_suppkey
                    AND l_shipdate >= DATE '1994-01-01'
                    AND l_shipdate <  DATE '1994-01-01' + INTERVAL '1 year'
              )
        )
          AND s_nationkey = n_nationkey
          AND n_name = 'CANADA'
        ORDER BY s_name
    """,

    "Q21": """
        SELECT s_name, COUNT(*) AS numwait
        FROM supplier, lineitem l1, orders, nation
        WHERE s_suppkey = l1.l_suppkey
          AND o_orderkey = l1.l_orderkey
          AND o_orderstatus = 'F'
          AND l1.l_receiptdate > l1.l_commitdate
          AND EXISTS (
              SELECT * FROM lineitem l2
              WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey
          )
          AND NOT EXISTS (
              SELECT * FROM lineitem l3
              WHERE l3.l_orderkey = l1.l_orderkey
                AND l3.l_suppkey <> l1.l_suppkey
                AND l3.l_receiptdate > l3.l_commitdate
          )
          AND s_nationkey = n_nationkey
          AND n_name = 'SAUDI ARABIA'
        GROUP BY s_name
        ORDER BY numwait DESC, s_name
        LIMIT 100
    """,
}


# ─── Phase Definitions ────────────────────────────────────────────────────────

# Phase A — LINEITEM/ORDERS heavy: revenue, shipment, order-priority analysis
PHASE_A = ["Q1", "Q3", "Q4", "Q6", "Q12"]

# Phase B — CUSTOMER/PART heavy: CRM, inventory, promotions
# (Q20 removed: nested correlated subqueries scan lineitem twice, ~109s at SF=0.2)
PHASE_B = ["Q10", "Q13", "Q14", "Q17", "Q18"]

# Phase C — Mixed analytical: complex cross-dimensional queries
# (Q9,Q21 removed: exceed timeout at SF≥1.0)
PHASE_C = ["Q2", "Q8", "Q11"]


# ─── Without-Replacement Phase Generator ─────────────────────────────────────

def _gen_phase(query_ids: list, n: int, rng: random.Random) -> list:
    """
    Generate a sequence of n query IDs using shuffled round-robin.

    Each round shuffles `query_ids` and appends the full shuffled list.
    This guarantees:
      • Every query type appears in each window of size len(query_ids) exactly once
        → P(intra-window type duplication) = 0% when window_size == len(query_ids)
      • Equal representation: each type appears exactly n // len(query_ids)
        times (plus at most one extra if n % len(query_ids) != 0)
      • No systematic ordering bias within phases

    For Phase C (3 types, window_size=5): two consecutive types appear in every
    window by pigeonhole — unavoidable, but minimised vs random choice (96%+).
    """
    result = []
    while len(result) < n:
        batch = query_ids[:]
        rng.shuffle(batch)
        result.extend(batch)
    return result[:n]


# ─── Workload Trace Builder ───────────────────────────────────────────────────

def get_workload_trace(
    queries_per_phase: int = 30,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    Generate a workload trace with deliberate phase shifts.

    Pattern: A×N → B×N → A×N → C×N

    Each item is (query_id, parameterized_sql_string).

    Parameters
    ----------
    queries_per_phase : int
        Number of queries to execute per phase.
        Default 30 → 120 total (4 phases × 30).
        With WINDOW_SIZE=5: 6 windows per phase, 24 windows total.
        Phase transitions at windows {6, 12, 18}.

    seed : int
        Controls both query ordering (within each phase) and parameter
        variant selection.  Pass seed=block_number so that each block
        uses a different parameter set → inter-block cache reuse is
        eliminated.

    Returns
    -------
    List of (query_id, sql) tuples.
    """
    rng = random.Random(seed)
    trace = []

    phases = [
        ("Phase_A",        PHASE_A),
        ("Phase_B",        PHASE_B),
        ("Phase_A_repeat", PHASE_A),
        ("Phase_C",        PHASE_C),
    ]

    for _phase_name, query_ids in phases:
        # Without-replacement ordering (Fix 1)
        qid_sequence = _gen_phase(
            [qid for qid in query_ids if qid in TPCH_TEMPLATES],
            queries_per_phase,
            rng,
        )
        for qid in qid_sequence:
            # Random parameter variant for this execution (Fix 2)
            params = rng.choice(QUERY_PARAM_POOLS.get(qid, [{}]))
            sql    = TPCH_TEMPLATES[qid].format(**params)
            trace.append((qid, sql))

    return trace


def get_window_queries(
    trace: List[Tuple[str, str]],
    start: int,
    window_size: int,
) -> List[str]:
    """Extract SQL strings for a window slice from the trace."""
    return [sql for _, sql in trace[start:start + window_size]]


def describe_trace(queries_per_phase: int = 30) -> None:
    """Print trace structure for verification."""
    from collections import Counter
    trace  = get_workload_trace(queries_per_phase, seed=42)
    total  = len(trace)
    qpp    = queries_per_phase
    window = 5   # WINDOW_SIZE assumed

    print(f"Workload trace: {total} queries  (seed=42)")
    print(f"  Phase A        (Q1,Q3,Q4,Q6,Q12)    : positions   0 – {qpp-1}")
    print(f"  Phase B        (Q10,Q13,Q14,Q17,Q18): positions {qpp:3d} – {2*qpp-1}")
    print(f"  Phase A repeat (Q1,Q3,Q4,Q6,Q12)    : positions {2*qpp:3d} – {3*qpp-1}")
    print(f"  Phase C        (Q2,Q8,Q11)           : positions {3*qpp:3d} – {4*qpp-1}")

    n_windows    = total // window
    n_per_phase  = qpp // window
    print(f"\nWindows (size={window}): {n_windows} total  ({n_per_phase} per phase)")
    print(f"Phase transitions at windows: "
          f"{{{n_per_phase}, {2*n_per_phase}, {3*n_per_phase}}}")

    print(f"\nIntra-window query-type diversity (first 4 windows):")
    for w in range(min(4, n_windows)):
        start = w * window
        ids   = [qid for qid, _ in trace[start:start + window]]
        dups  = len(ids) - len(set(ids))
        print(f"  Window {w}: {ids}  → {dups} duplicate type(s)")

    print(f"\nQuery-type distribution (full trace):")
    counts = Counter(qid for qid, _ in trace)
    for qid, cnt in sorted(counts.items()):
        print(f"  {qid:5s}: {cnt:3d}  ({cnt/total*100:.1f}%)")


if __name__ == "__main__":
    describe_trace(30)
    print("\nFirst 8 queries of trace (seed=0 vs seed=1):")
    for seed in [0, 1]:
        t = get_workload_trace(5, seed=seed)
        ids = [qid for qid, _ in t]
        print(f"  seed={seed}: {ids}")
