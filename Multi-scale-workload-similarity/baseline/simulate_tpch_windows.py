"""
simulate_tpch_windows.py
========================
Generates synthetic TPC-H workload windows matching the statistical
properties described in Paper 3A (31 windows, 3 phase changes at W7, W15, W23).

This is a REPRODUCIBLE SIMULATION for code-availability purposes.
Replace with actual TPC-H logs when available.

Phase structure (from paper):
  W1  – W7  : Phase 1 — heavy SELECT, mixed schema queries
  W8  – W15 : Phase 2 — INSERT/UPDATE surge (data loading)
  W16 – W23 : Phase 3 — analytical aggregation (ORDER BY, GROUP BY)
  W24 – W31 : Phase 4 — balanced mixed workload
"""

import numpy as np
from baseline_measures import WorkloadWindow

RNG = np.random.default_rng(seed=42)

# TPC-H tables and columns (simplified)
TABLES_PHASE = {
    1: {'orders','lineitem','customer'},
    2: {'orders','lineitem','supplier','partsupp'},
    3: {'lineitem','orders','part','partsupp'},
    4: {'customer','orders','lineitem','nation'},
}
COLS_PHASE = {
    1: {'o_orderkey','l_linenumber','c_custkey','l_quantity'},
    2: {'o_orderkey','l_suppkey','ps_partkey','s_suppkey','l_quantity'},
    3: {'l_partkey','p_partkey','ps_suppkey','l_extendedprice'},
    4: {'c_custkey','o_custkey','l_orderkey','n_nationkey'},
}


def phase_of(window_id: int) -> int:
    if   window_id <= 7:  return 1
    elif window_id <= 15: return 2
    elif window_id <= 23: return 3
    else:                 return 4


def make_query_type_vec(phase: int) -> np.ndarray:
    """[frac_select, frac_insert, frac_update, frac_delete]"""
    base = {
        1: [0.80, 0.10, 0.05, 0.05],
        2: [0.35, 0.45, 0.15, 0.05],
        3: [0.90, 0.05, 0.03, 0.02],
        4: [0.60, 0.20, 0.12, 0.08],
    }[phase]
    noise = RNG.normal(0, 0.02, 4)
    vec = np.clip(np.array(base) + noise, 0, 1)
    return vec / vec.sum()


def make_temporal_bands(phase: int, n_pts: int = 19) -> dict:
    """Simulate DWT band coefficients for a workload window."""
    # Phase-specific temporal patterns
    t = np.linspace(0, 2 * np.pi, n_pts)
    freq_map = {1: 1.0, 2: 2.5, 3: 0.5, 4: 1.5}
    amp_map  = {1: 1.0, 2: 2.0, 3: 0.7, 4: 1.2}
    freq = freq_map[phase]; amp = amp_map[phase]
    signal = amp * np.sin(freq * t) + RNG.normal(0, 0.1, n_pts)
    # Simple DWT approximation (just use windowed averages as proxy)
    def avg_band(sig, scale):
        k = max(1, len(sig) // scale)
        return np.array([sig[i*k:(i+1)*k].mean() for i in range(scale)])
    return {
        'cA3': avg_band(signal, 2) + RNG.normal(0, 0.05, 2),
        'cD3': avg_band(signal, 4) + RNG.normal(0, 0.08, 4),
        'cD2': avg_band(signal, 4) + RNG.normal(0, 0.10, 4),
        'cD1': avg_band(signal, 8) + RNG.normal(0, 0.12, 8),
    }


def make_sax(phase: int, w: int = 8, alpha: int = 4) -> np.ndarray:
    """Simplified SAX representation."""
    return RNG.integers(0, alpha, size=w)


QPS_MAP = {1: 18.2, 2: 9.1, 3: 22.5, 4: 15.0}


def generate_windows(n_windows: int = 31) -> list:
    windows = []
    for wid in range(1, n_windows + 1):
        ph = phase_of(wid)
        qps = QPS_MAP[ph] * (1 + RNG.normal(0, 0.05))
        sr  = make_query_type_vec(ph)[0]   # select fraction
        qt  = make_query_type_vec(ph)
        # Add random table/col noise for transitions
        tbls = set(TABLES_PHASE[ph])
        cols = set(COLS_PHASE[ph])
        if RNG.random() < 0.15:            # 15% chance of extra table
            extra = {'region', 'nation', 'part'}
            tbls |= {RNG.choice(list(extra))}
        w = WorkloadWindow(
            window_id      = wid,
            qps            = max(1.0, qps),
            select_ratio   = float(np.clip(sr, 0, 1)),
            query_type_vec = qt,
            table_set      = tbls,
            col_set        = cols,
            temporal_sax   = make_sax(ph),
            temporal_bands = make_temporal_bands(ph),
        )
        windows.append(w)
    return windows


if __name__ == '__main__':
    ws = generate_windows(31)
    print(f"Generated {len(ws)} windows")
    for w in ws[:5]:
        print(f"  W{w.window_id:02d} phase={phase_of(w.window_id)} "
              f"QPS={w.qps:.1f} SR={w.select_ratio:.2f} "
              f"tables={sorted(w.table_set)}")
