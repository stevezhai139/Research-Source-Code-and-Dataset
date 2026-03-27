"""
hsm/windowing.py
================
Converts raw TPC-H query trace (trace.csv) into WorkloadWindow objects
suitable for HSM computation.

Input CSV format (from STEP2_run_workload.py):
  run, window, phase, seq, query, op_type, exec_ms, ok, err,
  planning_ms, exec_ms_pg, rows_returned, shared_blks_hit, shared_blks_read,
  temp_blks_written

Output: List[WorkloadWindow] per run, with full feature extraction.

TPC-H column access map (12 representative queries):
  Q1, Q3, Q4, Q5, Q6, Q7, Q10, Q11, Q12, Q14, Q17, Q18
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from .measures import WorkloadWindow

# ─── TPC-H column access map ─────────────────────────────────────────────────

QUERY_COLS: Dict[str, set] = {
    # ── Long-form names (from STEP3_compute_hsm.py) ──────────────────
    "Q1_pricing_summary": {
        "lineitem.l_shipdate", "lineitem.l_returnflag", "lineitem.l_linestatus",
        "lineitem.l_quantity", "lineitem.l_extendedprice",
        "lineitem.l_discount", "lineitem.l_tax",
    },
    "Q3_shipping_priority": {
        "customer.c_mktsegment", "customer.c_custkey",
        "orders.o_custkey", "orders.o_orderdate", "orders.o_shippriority",
        "lineitem.l_orderkey", "lineitem.l_shipdate",
        "lineitem.l_extendedprice", "lineitem.l_discount",
    },
    "Q4_order_priority": {
        "orders.o_orderdate", "orders.o_orderpriority", "orders.o_orderkey",
        "lineitem.l_orderkey", "lineitem.l_commitdate", "lineitem.l_receiptdate",
    },
    "Q5_local_revenue": {
        "customer.c_custkey", "customer.c_nationkey",
        "orders.o_custkey", "orders.o_orderdate",
        "lineitem.l_orderkey", "lineitem.l_suppkey",
        "lineitem.l_extendedprice", "lineitem.l_discount",
        "supplier.s_suppkey", "supplier.s_nationkey",
        "nation.n_nationkey", "nation.n_name",
        "region.r_regionkey", "region.r_name",
    },
    "Q6_revenue_change": {
        "lineitem.l_shipdate", "lineitem.l_discount",
        "lineitem.l_quantity", "lineitem.l_extendedprice",
    },
    "Q7_volume_shipping": {
        "supplier.s_suppkey", "supplier.s_nationkey",
        "lineitem.l_suppkey", "lineitem.l_orderkey",
        "lineitem.l_shipdate", "lineitem.l_extendedprice", "lineitem.l_discount",
        "orders.o_orderkey", "orders.o_custkey",
        "customer.c_custkey", "customer.c_nationkey",
        "nation.n_nationkey", "nation.n_name",
    },
    "Q10_returned_items": {
        "customer.c_custkey", "customer.c_name", "customer.c_acctbal",
        "customer.c_nationkey", "orders.o_custkey", "orders.o_orderdate",
        "orders.o_orderkey", "lineitem.l_orderkey", "lineitem.l_returnflag",
        "lineitem.l_extendedprice", "lineitem.l_discount",
        "nation.n_nationkey", "nation.n_name",
    },
    "Q11_supply_cost": {
        "partsupp.ps_suppkey", "partsupp.ps_supplycost",
        "partsupp.ps_availqty", "partsupp.ps_partkey",
        "supplier.s_suppkey", "supplier.s_nationkey",
        "nation.n_nationkey", "nation.n_name",
    },
    "Q12_shipping_mode": {
        "orders.o_orderkey", "orders.o_orderpriority",
        "lineitem.l_orderkey", "lineitem.l_shipmode",
        "lineitem.l_commitdate", "lineitem.l_receiptdate", "lineitem.l_shipdate",
    },
    "Q14_promo_revenue": {
        "lineitem.l_partkey", "lineitem.l_shipdate",
        "lineitem.l_extendedprice", "lineitem.l_discount",
        "part.p_partkey", "part.p_type",
    },
    "Q17_small_quantity": {
        "lineitem.l_partkey", "lineitem.l_quantity", "lineitem.l_extendedprice",
        "part.p_partkey", "part.p_brand", "part.p_container",
    },
    "Q18_large_volume": {
        "customer.c_name", "customer.c_custkey",
        "orders.o_orderkey", "orders.o_custkey",
        "orders.o_orderdate", "orders.o_totalprice",
        "lineitem.l_orderkey", "lineitem.l_quantity",
    },
    # ── Short-form names (from actual trace CSVs) ─────────────────────
    "Q1":  {"lineitem.l_shipdate", "lineitem.l_returnflag", "lineitem.l_linestatus",
             "lineitem.l_quantity", "lineitem.l_extendedprice", "lineitem.l_discount",
             "lineitem.l_tax"},
    "Q3":  {"customer.c_mktsegment", "customer.c_custkey",
             "orders.o_custkey", "orders.o_orderdate", "orders.o_shippriority",
             "lineitem.l_orderkey", "lineitem.l_shipdate",
             "lineitem.l_extendedprice", "lineitem.l_discount"},
    "Q4":  {"orders.o_orderdate", "orders.o_orderpriority", "orders.o_orderkey",
             "lineitem.l_orderkey", "lineitem.l_commitdate", "lineitem.l_receiptdate"},
    "Q5":  {"customer.c_custkey", "customer.c_nationkey",
             "orders.o_custkey", "orders.o_orderdate",
             "lineitem.l_orderkey", "lineitem.l_suppkey",
             "lineitem.l_extendedprice", "lineitem.l_discount",
             "supplier.s_suppkey", "supplier.s_nationkey",
             "nation.n_nationkey", "nation.n_name",
             "region.r_regionkey", "region.r_name"},
    "Q6":  {"lineitem.l_shipdate", "lineitem.l_discount",
             "lineitem.l_quantity", "lineitem.l_extendedprice"},
    "Q7":  {"supplier.s_suppkey", "supplier.s_nationkey",
             "lineitem.l_suppkey", "lineitem.l_orderkey",
             "lineitem.l_shipdate", "lineitem.l_extendedprice", "lineitem.l_discount",
             "orders.o_orderkey", "orders.o_custkey",
             "customer.c_custkey", "customer.c_nationkey",
             "nation.n_nationkey", "nation.n_name"},
    "Q10": {"customer.c_custkey", "customer.c_name", "customer.c_acctbal",
             "customer.c_nationkey", "orders.o_custkey", "orders.o_orderdate",
             "orders.o_orderkey", "lineitem.l_orderkey", "lineitem.l_returnflag",
             "lineitem.l_extendedprice", "lineitem.l_discount",
             "nation.n_nationkey", "nation.n_name"},
    "Q11": {"partsupp.ps_suppkey", "partsupp.ps_supplycost",
             "partsupp.ps_availqty", "partsupp.ps_partkey",
             "supplier.s_suppkey", "supplier.s_nationkey",
             "nation.n_nationkey", "nation.n_name"},
    "Q12": {"orders.o_orderkey", "orders.o_orderpriority",
             "lineitem.l_orderkey", "lineitem.l_shipmode",
             "lineitem.l_commitdate", "lineitem.l_receiptdate", "lineitem.l_shipdate"},
    "Q14": {"lineitem.l_partkey", "lineitem.l_shipdate",
             "lineitem.l_extendedprice", "lineitem.l_discount",
             "part.p_partkey", "part.p_type"},
    "Q17": {"lineitem.l_partkey", "lineitem.l_quantity", "lineitem.l_extendedprice",
             "part.p_partkey", "part.p_brand", "part.p_container"},
    "Q18": {"customer.c_name", "customer.c_custkey",
             "orders.o_orderkey", "orders.o_custkey",
             "orders.o_orderdate", "orders.o_totalprice",
             "lineitem.l_orderkey", "lineitem.l_quantity"},
    # ── DML queries (INSERT/UPDATE/DELETE) ────────────────────────────
    "I_customer": {"customer.c_custkey", "customer.c_name", "customer.c_address",
                   "customer.c_nationkey", "customer.c_phone", "customer.c_acctbal"},
    "U_orders":   {"orders.o_orderkey", "orders.o_orderstatus", "orders.o_totalprice",
                   "orders.o_orderdate"},
    "U_customer": {"customer.c_custkey", "customer.c_acctbal", "customer.c_phone"},
    "D_orders":   {"orders.o_orderkey", "orders.o_custkey"},
    "I_orders":   {"orders.o_orderkey", "orders.o_custkey", "orders.o_orderstatus",
                   "orders.o_totalprice", "orders.o_orderdate"},
    "D_customer": {"customer.c_custkey"},
}

# Derive table sets from column sets
QUERY_TABLES: Dict[str, set] = {
    q: {c.split('.')[0] for c in cols}
    for q, cols in QUERY_COLS.items()
}

# Will be built dynamically per dataset — set populated in load_trace()
ALL_QUERY_NAMES: List[str] = sorted(QUERY_COLS.keys())


# ─── Core extraction functions ────────────────────────────────────────────────

def _query_type_vec(win_df: pd.DataFrame,
                    all_query_names: Optional[List[str]] = None) -> np.ndarray:
    """
    Build normalised frequency vector over all known query names.
    Used for S_T (cosine similarity on query-type distribution).

    Uses the provided all_query_names (built from the full dataset) to ensure
    consistent dimensionality across all windows.
    """
    if all_query_names is None:
        all_query_names = ALL_QUERY_NAMES
    counts = win_df['query'].value_counts()
    total  = max(len(win_df), 1)
    vec = np.array([counts.get(q, 0) / total for q in all_query_names], dtype=float)
    s = vec.sum()
    return vec / s if s > 0 else vec


def _query_rank_vec(win_df: pd.DataFrame,
                    all_query_names: Optional[List[str]] = None) -> np.ndarray:
    """
    Rank-based query frequency vector used for S_R (Spearman correlation).
    Same as _query_type_vec — Spearman rank-correlation handles the ordering.
    """
    return _query_type_vec(win_df, all_query_names)


def _table_col_sets(win_df: pd.DataFrame):
    """Extract union of tables and columns accessed in the window."""
    tables: set = set()
    cols:   set = set()
    for q in win_df['query'].unique():
        tables |= QUERY_TABLES.get(q, set())
        cols   |= QUERY_COLS.get(q, set())
    return tables, cols


def _temporal_bands_from_arrival(win_df: pd.DataFrame,
                                  n_pts: int = 19,
                                  alpha: int = 4) -> dict:
    """
    Compute DWT band proxies from intra-window query-arrival sequence.

    We use the arrival timestamps (seq_in_win) within the window to build
    a density histogram, then apply a 3-level Haar DWT approximation.

    This gives a deterministic S_P that captures temporal burstiness —
    matching the mathematical definition in Theorem 3 (wavelet optimality).
    """
    import pywt

    seq = win_df['seq'].values.astype(float)
    if len(seq) == 0:
        zero = np.zeros(n_pts)
        return {'cA3': zero[:2], 'cD3': zero[:4], 'cD2': zero[:4], 'cD1': zero[:8]}

    # Normalise to [0, n_pts - 1] and build density signal
    total = max(seq.max() - seq.min(), 1.0)
    bins  = np.clip(((seq - seq.min()) / total * (n_pts - 1)).astype(int), 0, n_pts - 1)
    signal = np.bincount(bins, minlength=n_pts).astype(float)

    # Pad to next power-of-2 for proper DWT
    padlen = 2 ** int(np.ceil(np.log2(len(signal) + 1)))
    sig_pad = np.pad(signal, (0, padlen - len(signal)), mode='constant')

    try:
        coeffs = pywt.wavedec(sig_pad, 'haar', level=3)
        cA3 = coeffs[0]
        cD3 = coeffs[1]
        cD2 = coeffs[2]
        cD1 = coeffs[3] if len(coeffs) > 3 else coeffs[2]
    except Exception:
        # Fallback: simple windowed averages
        k = max(1, n_pts // 2)
        def avg_band(sig, s):
            return np.array([sig[i*s:(i+1)*s].mean() for i in range(s)])
        cA3 = avg_band(signal, 2)
        cD3 = avg_band(signal, 4)
        cD2 = avg_band(signal, 4)
        cD1 = avg_band(signal, 8)

    return {'cA3': cA3, 'cD3': cD3, 'cD2': cD2, 'cD1': cD1}


def window_from_df(wid: int, phase: str, win_df: pd.DataFrame,
                   window_duration_s: float = 60.0,
                   all_query_names: Optional[List[str]] = None) -> WorkloadWindow:
    """
    Build a WorkloadWindow from a DataFrame slice for one window.

    Args:
        wid:               Window ID (int)
        phase:             Phase label string
        win_df:            Rows belonging to this window
        window_duration_s: Duration of one window in seconds (for QPS calculation)
    """
    n = len(win_df)
    qps = n / window_duration_s if window_duration_s > 0 else float(n)

    qt_vec   = _query_type_vec(win_df, all_query_names)
    rank_vec = _query_rank_vec(win_df, all_query_names)
    tables, cols = _table_col_sets(win_df)

    # select_ratio from op_type column
    if 'op_type' in win_df.columns:
        n_select = (win_df['op_type'].str.upper() == 'SELECT').sum()
    else:
        n_select = n  # assume all SELECT if column absent
    select_ratio = float(n_select / max(n, 1))

    try:
        bands = _temporal_bands_from_arrival(win_df)
    except Exception:
        bands = {}

    # SAX representation: 8-symbol discretisation of arrival signal
    sax_len = 8
    if n > 0:
        seq = win_df['seq'].values.astype(float)
        total = max(seq.max() - seq.min(), 1.0)
        norm  = (seq - seq.min()) / total
        sax   = (norm * sax_len).clip(0, sax_len - 1).astype(int)[:sax_len]
        if len(sax) < sax_len:
            sax = np.pad(sax, (0, sax_len - len(sax)), mode='edge')
    else:
        sax = np.zeros(sax_len, dtype=int)

    return WorkloadWindow(
        window_id      = wid,
        phase          = phase,
        qps            = max(qps, 0.01),
        select_ratio   = select_ratio,
        query_type_vec = qt_vec,
        query_rank_vec = rank_vec,
        table_set      = tables,
        col_set        = cols,
        temporal_sax   = sax,
        temporal_bands = bands,
        n_queries      = n,
    )


# ─── Load a complete run from trace CSV ──────────────────────────────────────

def load_trace(trace_path: str, run_id: int = 1,
               window_duration_s: float = 60.0) -> List[WorkloadWindow]:
    """
    Load all workload windows from a trace.csv file.

    Args:
        trace_path:        Path to trace.csv
        run_id:            Which run number to extract (if multi-run file)
        window_duration_s: Assumed window duration in seconds

    Returns:
        List of WorkloadWindow objects, sorted by window_id.
    """
    df = pd.read_csv(trace_path)

    # Filter to successful queries and requested run
    if 'ok' in df.columns:
        df = df[df['ok'].astype(str).str.lower() == 'true']
    if 'run' in df.columns:
        df = df[df['run'] == run_id]

    # Normalise column names
    df = df.rename(columns={'window': 'window_id', 'query': 'query',
                             'seq': 'seq', 'phase': 'phase'})

    # Build vocabulary of all query names observed in THIS dataset
    # (ensures consistent feature vectors regardless of trace content)
    observed_queries = sorted(df['query'].unique())
    all_query_names  = observed_queries  # data-driven vocabulary

    windows = []
    for wid in sorted(df['window_id'].unique()):
        win_df = df[df['window_id'] == wid].reset_index(drop=True)
        phase  = win_df['phase'].iloc[0] if 'phase' in win_df.columns else 'Unknown'
        w      = window_from_df(int(wid), str(phase), win_df,
                                window_duration_s, all_query_names)
        windows.append(w)

    return windows


def load_all_runs(results_dir: str,
                  window_duration_s: float = 60.0) -> Dict[int, List[WorkloadWindow]]:
    """
    Load all run_XX/trace.csv files from a results directory.

    Args:
        results_dir: Path like 'data/tpch_sf0.2/' containing run_01/, run_02/, ...

    Returns:
        Dict mapping run_id → List[WorkloadWindow]
    """
    results_path = Path(results_dir)
    all_runs = {}

    # Support both multi-run trace (single file with run column) and
    # per-run subdirectories (run_01/trace.csv)
    combined = results_path / 'trace.csv'
    if combined.exists():
        df_all = pd.read_csv(combined)
        if 'run' in df_all.columns:
            for run_id in sorted(df_all['run'].unique()):
                all_runs[run_id] = load_trace(str(combined), run_id, window_duration_s)
        else:
            all_runs[1] = load_trace(str(combined), 1, window_duration_s)
        return all_runs

    for run_dir in sorted(results_path.glob('run_*')):
        trace_file = run_dir / 'trace.csv'
        if not trace_file.exists():
            continue
        try:
            run_id = int(run_dir.name.split('_')[1])
        except (ValueError, IndexError):
            continue
        df = pd.read_csv(trace_file)
        # Add synthetic run column if missing
        if 'run' not in df.columns:
            df['run'] = run_id
        df.to_csv(trace_file, index=False)  # back-patch
        all_runs[run_id] = load_trace(str(trace_file), run_id, window_duration_s)

    return all_runs
