"""
baseline_measures.py
====================
Implements simple workload similarity baselines for comparison with HSM.
Baselines intentionally use ONLY single or partial dimensions to show that
HSM's 5-dimensional design is superior.

Baseline ID  Description                          Dimensions Used
-----------  -----------------------------------  ----------------
B0           Euclidean on (QPS, select_ratio)     ~S_R + S_V partial
B1           Cosine on query-type vector only     S_T only
B2           Jaccard on table sets only           S_A only
B3           HSM-2D: S_R + S_V                   2 / 5 dims
B4           HSM-3D: S_R + S_V + S_T             3 / 5 dims
B5           HSM-4D: S_R + S_V + S_T + S_A       4 / 5 dims
B6           HSM-5D (full)                        5 / 5 dims  ← proposed
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class WorkloadWindow:
    """Feature representation of a workload window."""
    window_id:    int
    qps:          float           # queries per second
    select_ratio: float           # fraction of SELECT queries in [0,1]
    query_type_vec: np.ndarray    # 4-dim: [frac_select, frac_insert, frac_update, frac_delete]
    table_set:    Set[str]        # tables accessed
    col_set:      Set[str]        # columns accessed
    temporal_sax: np.ndarray      # SAX representation of q(t) (length w, alphabet α)
    temporal_bands: dict          # {'cA3': np.array, 'cD3':..., 'cD2':..., 'cD1':...}


# ─── Individual similarity functions ─────────────────────────────────────────

def sim_SR(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """Rate similarity: SELECT/non-SELECT ratio distance."""
    ratio_a = wa.select_ratio
    ratio_b = wb.select_ratio
    return 1.0 - abs(ratio_a - ratio_b)


def sim_SV(wa: WorkloadWindow, wb: WorkloadWindow, eps: float = 1e-9) -> float:
    """Volume similarity: log-scale QPS ratio."""
    log_diff = abs(math.log10(wa.qps + eps) - math.log10(wb.qps + eps))
    return math.exp(-log_diff)


def sim_ST(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """Type similarity: cosine similarity on query-type vectors."""
    va, vb = wa.query_type_vec, wb.query_type_vec
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-12:
        return 1.0
    return float(np.dot(va, vb) / denom)


def sim_SA(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """Access-attribute similarity: Jaccard on table + column sets."""
    t_inter = len(wa.table_set & wb.table_set)
    t_union = len(wa.table_set | wb.table_set)
    c_inter = len(wa.col_set & wb.col_set)
    c_union = len(wa.col_set | wb.col_set)
    j_t = t_inter / t_union if t_union > 0 else 1.0
    j_c = c_inter / c_union if c_union > 0 else 1.0
    return 0.5 * j_t + 0.5 * j_c


def sim_SP_band(wa: WorkloadWindow, wb: WorkloadWindow,
                band: str, window_len: int, radius: int = 3,
                alpha: int = 4) -> float:
    """Pattern similarity for one DWT band using FastDTW approximation."""
    sa = wa.temporal_bands[band]
    sb = wb.temporal_bands[band]
    # Use numpy-based DTW with Sakoe-Chiba band
    n = len(sa)
    INF = float('inf')
    # Banded DTW
    D = np.full((n+1, n+1), INF)
    D[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(max(1, i-radius), min(n+1, i+radius+1)):
            cost = abs(sa[i-1] - sb[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    dtw_dist = D[n, n]
    denom = window_len * (alpha - 1)
    if denom <= 0:
        return 1.0
    return max(0.0, 1.0 - dtw_dist / denom)


def sim_SP(wa: WorkloadWindow, wb: WorkloadWindow,
           window_len: int = 19, radius: int = 3,
           weights: dict = None) -> float:
    """Pattern similarity: weighted sum over DWT bands."""
    if weights is None:
        weights = {'cA3': 0.4, 'cD3': 0.2, 'cD2': 0.2, 'cD1': 0.2}
    sp = 0.0
    for band, lam in weights.items():
        sp += lam * sim_SP_band(wa, wb, band, window_len, radius)
    return sp


# ─── Baseline measures ───────────────────────────────────────────────────────

def baseline_B0_euclidean(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """B0: L2 distance on (normalised QPS, select_ratio). Converted to [0,1]."""
    eps = 1e-9
    qps_max = max(wa.qps, wb.qps, eps)
    v_a = np.array([wa.qps / qps_max, wa.select_ratio])
    v_b = np.array([wb.qps / qps_max, wb.select_ratio])
    dist = np.linalg.norm(v_a - v_b)
    return max(0.0, 1.0 - dist / math.sqrt(2))


def baseline_B1_cosine_qt(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """B1: Cosine similarity on query-type vector only (= S_T)."""
    return sim_ST(wa, wb)


def baseline_B2_jaccard_schema(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """B2: Jaccard on table sets only (partial S_A)."""
    t_inter = len(wa.table_set & wb.table_set)
    t_union = len(wa.table_set | wb.table_set)
    return t_inter / t_union if t_union > 0 else 1.0


def baseline_B3_HSM2D(wa: WorkloadWindow, wb: WorkloadWindow,
                       w_R=0.5, w_V=0.5) -> float:
    """B3: HSM with only S_R + S_V (no type, schema, pattern)."""
    return w_R * sim_SR(wa, wb) + w_V * sim_SV(wa, wb)


def baseline_B4_HSM3D(wa: WorkloadWindow, wb: WorkloadWindow,
                       w_R=0.33, w_V=0.33, w_T=0.34) -> float:
    """B4: HSM with S_R + S_V + S_T (no schema, no pattern)."""
    return w_R * sim_SR(wa, wb) + w_V * sim_SV(wa, wb) + w_T * sim_ST(wa, wb)


def baseline_B5_HSM4D(wa: WorkloadWindow, wb: WorkloadWindow,
                       w_R=0.25, w_V=0.25, w_T=0.25, w_A=0.25) -> float:
    """B5: HSM with S_R + S_V + S_T + S_A (no temporal pattern)."""
    return (w_R * sim_SR(wa, wb) + w_V * sim_SV(wa, wb)
          + w_T * sim_ST(wa, wb) + w_A * sim_SA(wa, wb))


def baseline_B6_HSM5D(wa: WorkloadWindow, wb: WorkloadWindow,
                       weights: dict = None) -> float:
    """B6: Full HSM (proposed method)."""
    if weights is None:
        weights = {'R': 0.20, 'V': 0.20, 'T': 0.20, 'A': 0.20, 'P': 0.20}
    return (weights['R'] * sim_SR(wa, wb)
          + weights['V'] * sim_SV(wa, wb)
          + weights['T'] * sim_ST(wa, wb)
          + weights['A'] * sim_SA(wa, wb)
          + weights['P'] * sim_SP(wa, wb))


# ─── Evaluation ──────────────────────────────────────────────────────────────

ALL_BASELINES = {
    'B0-Euclidean':    baseline_B0_euclidean,
    'B1-Cosine-QT':    baseline_B1_cosine_qt,
    'B2-Jaccard-Sch':  baseline_B2_jaccard_schema,
    'B3-HSM-2D':       baseline_B3_HSM2D,
    'B4-HSM-3D':       baseline_B4_HSM3D,
    'B5-HSM-4D':       baseline_B5_HSM4D,
    'B6-HSM-5D':       baseline_B6_HSM5D,
}


def evaluate_change_detection(windows: List[WorkloadWindow],
                               true_change_points: List[int],
                               theta: float = 0.75,
                               ref_lag: int = 1) -> dict:
    """
    Evaluate similarity measures on workload drift detection.

    Strategy: compare each window W_i against W_{i-ref_lag}.
    A 'change' is detected when sim(W_i, W_{i-1}) < theta.

    Returns precision, recall, F1 for each baseline.
    """
    results = {}
    n = len(windows)
    true_set = set(true_change_points)

    for name, sim_fn in ALL_BASELINES.items():
        scores = []
        predicted = []
        for i in range(ref_lag, n):
            s = sim_fn(windows[i-ref_lag], windows[i])
            scores.append(s)
            if s < theta:
                predicted.append(i)

        pred_set = set(predicted)
        # Allow ±1 window tolerance for change point matching
        tp = sum(1 for p in pred_set
                 if any(abs(p - t) <= 1 for t in true_set))
        fp = len(pred_set) - tp
        fn = len(true_set) - sum(1 for t in true_set
                                 if any(abs(p - t) <= 1 for p in pred_set))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        results[name] = {
            'scores': scores,
            'predicted_changes': sorted(predicted),
            'precision': round(precision, 3),
            'recall':    round(recall, 3),
            'f1':        round(f1, 3),
        }
    return results


def print_comparison_table(results: dict) -> None:
    """Print LaTeX-ready comparison table."""
    header = f"{'Method':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Changes'}"
    print(header)
    print('-' * 65)
    for name, r in results.items():
        marker = ' ← proposed' if 'HSM-5D' in name else ''
        print(f"{name:<20} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['f1']:>8.3f}  {r['predicted_changes']}{marker}")
