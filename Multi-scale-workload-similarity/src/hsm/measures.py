"""
hsm/measures.py
===============
Implements all 7 workload similarity measures evaluated in Paper 3A.

Measure hierarchy (from simplest to full HSM):

  ID   | Name             | Dimensions         | Paper ref
  -----|------------------|--------------------|----------
  B0   | Volume-Ratio     | S_V only (1/5)     | Baseline
  B1   | Cosine-QT        | S_T only (1/5)     | Baseline
  B2   | Jaccard-Schema   | S_A only (1/5)     | Baseline
  B3   | HSM-2D           | S_R + S_V (2/5)    | Ablation
  B4   | HSM-3D           | S_R + S_V + S_T    | Ablation
  B5   | HSM-4D (no S_P)  | S_R+S_V+S_T+S_A   | Ablation
  B6   | HSM-5D (full)    | All 5 dims         | Proposed

Reference:
  Theorem 6 (Dimensional Necessity): Removing any single dimension
  causes measurable loss of discrimination power.
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Set, Optional


# ─── WorkloadWindow ───────────────────────────────────────────────────────────

@dataclass
class WorkloadWindow:
    """
    Feature representation of a single workload window W_i.

    Fields correspond directly to the five HSM dimensions:
      S_R  ← select_ratio / query_rank_vec
      S_V  ← qps
      S_T  ← query_type_vec
      S_A  ← table_set, col_set
      S_P  ← temporal_sax, temporal_bands
    """
    window_id:      int
    phase:          str                # Phase label (e.g., 'Reporting')
    qps:            float              # Queries per second
    select_ratio:   float              # Fraction of SELECT in [0,1]
    query_type_vec: np.ndarray         # [frac_select, frac_insert, frac_update, frac_delete]
    query_rank_vec: np.ndarray         # Spearman-rank distribution over query names
    table_set:      Set[str]           # Tables accessed in this window
    col_set:        Set[str]           # Columns accessed in this window
    temporal_sax:   np.ndarray         # SAX symbols (length w)
    temporal_bands: dict               # {'cA3': arr, 'cD3': arr, 'cD2': arr, 'cD1': arr}
    n_queries:      int   = 0          # Raw query count in window


# ─── Dimension-level similarity functions ────────────────────────────────────

def sim_SR(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """
    S_R — Rate similarity.
    Spearman rank correlation on query-type frequency vectors, scaled to [0,1].
    """
    from scipy.stats import spearmanr
    v1, v2 = wa.query_rank_vec, wb.query_rank_vec
    if len(v1) == 0 or len(v2) == 0 or v1.sum() == 0 or v2.sum() == 0:
        return 0.5
    r, _ = spearmanr(v1, v2)
    if np.isnan(r):
        r = 1.0
    return float((r + 1.0) / 2.0)


def sim_SV(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """
    S_V — Volume similarity.
    Min/Max ratio of query counts, mapped to [0,1].
    """
    n1, n2 = wa.n_queries, wb.n_queries
    if max(n1, n2) == 0:
        return 1.0
    return float(min(n1, n2) / max(n1, n2))


def sim_ST(wa: WorkloadWindow, wb: WorkloadWindow, n_slots: int = 10) -> float:
    """
    S_T — Type / temporal-arrival similarity.
    Cosine similarity on intra-window query-arrival histograms.
    """
    va, vb = wa.query_type_vec, wb.query_type_vec
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-12:
        return 1.0
    return float(np.dot(va, vb) / denom)


def sim_SA(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """
    S_A — Access-attribute similarity.
    Weighted Jaccard (0.5 table + 0.5 column) ∈ [0,1].
    """
    t_inter = len(wa.table_set & wb.table_set)
    t_union = len(wa.table_set | wb.table_set)
    c_inter = len(wa.col_set & wb.col_set)
    c_union = len(wa.col_set | wb.col_set)
    j_t = t_inter / t_union if t_union > 0 else 1.0
    j_c = c_inter / c_union if c_union > 0 else 1.0
    return 0.5 * j_t + 0.5 * j_c


def _dtw_band(sa: np.ndarray, sb: np.ndarray, radius: int, alpha: int) -> float:
    """Sakoe-Chiba banded DTW distance, normalised to [0,1]."""
    n = len(sa)
    if n == 0:
        return 0.0
    INF = float('inf')
    D = np.full((n + 1, n + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - radius), min(n + 1, i + radius + 1)):
            cost = abs(sa[i-1] - sb[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    dtw_dist = D[n, n]
    denom = n * (alpha - 1)
    return max(0.0, 1.0 - dtw_dist / denom) if denom > 0 else 1.0


def sim_SP(wa: WorkloadWindow, wb: WorkloadWindow,
           radius: int = 3, alpha: int = 4,
           weights: Optional[dict] = None) -> float:
    """
    S_P — Pattern similarity.
    Weighted FastDTW over DWT bands: cA3(0.4) + cD3(0.2) + cD2(0.2) + cD1(0.2).
    Falls back to query-type Jaccard if temporal bands are empty.
    """
    if weights is None:
        weights = {'cA3': 0.4, 'cD3': 0.2, 'cD2': 0.2, 'cD1': 0.2}
    # If using pre-computed temporal_bands (from synthetic or real DWT)
    if wa.temporal_bands and wb.temporal_bands:
        sp = 0.0
        for band, lam in weights.items():
            sa = wa.temporal_bands.get(band, np.array([]))
            sb = wb.temporal_bands.get(band, np.array([]))
            if len(sa) > 0 and len(sa) == len(sb):
                sp += lam * _dtw_band(sa, sb, radius, alpha)
        return sp
    # Fallback: query-type set Jaccard (used when trace-based windowing)
    t1 = wa.table_set   # repurposed as query-type set in trace mode
    t2 = wb.table_set
    union = t1 | t2
    return float(len(t1 & t2) / len(union)) if union else 1.0


# ─── HSM score (weighted sum) ─────────────────────────────────────────────────

DEFAULT_WEIGHTS = {'R': 0.25, 'V': 0.20, 'T': 0.20, 'A': 0.20, 'P': 0.15}


def hsm_score(wa: WorkloadWindow, wb: WorkloadWindow,
              weights: Optional[dict] = None) -> dict:
    """
    Compute full 5-D HSM score and all component scores.
    Returns dict with keys: S_R, S_V, S_T, S_A, S_P, HSM.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    sr = sim_SR(wa, wb)
    sv = sim_SV(wa, wb)
    st = sim_ST(wa, wb)
    sa = sim_SA(wa, wb)
    sp = sim_SP(wa, wb)
    hsm = (weights['R'] * sr + weights['V'] * sv + weights['T'] * st
           + weights['A'] * sa + weights['P'] * sp)
    return {
        'S_R': round(sr, 4), 'S_V': round(sv, 4),
        'S_T': round(st, 4), 'S_A': round(sa, 4),
        'S_P': round(sp, 4), 'HSM': round(float(hsm), 4),
    }


# ─── Baseline measures (B0–B6) ────────────────────────────────────────────────

def B0_volume_ratio(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """B0: Volume ratio only (S_V). Simplest numeric baseline."""
    return sim_SV(wa, wb)


def B1_cosine_qt(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """B1: Cosine similarity on query-type vector (= S_T)."""
    return sim_ST(wa, wb)


def B2_jaccard_schema(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """B2: Jaccard on accessed table sets only (partial S_A)."""
    t_inter = len(wa.table_set & wb.table_set)
    t_union = len(wa.table_set | wb.table_set)
    return t_inter / t_union if t_union > 0 else 1.0


def B3_HSM_2D(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """B3: HSM with S_R + S_V only."""
    return 0.5 * sim_SR(wa, wb) + 0.5 * sim_SV(wa, wb)


def B4_HSM_3D(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """B4: HSM with S_R + S_V + S_T."""
    return (sim_SR(wa, wb) + sim_SV(wa, wb) + sim_ST(wa, wb)) / 3.0


def B5_HSM_4D(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """B5: HSM with S_R + S_V + S_T + S_A (no temporal pattern S_P)."""
    return (sim_SR(wa, wb) + sim_SV(wa, wb) + sim_ST(wa, wb) + sim_SA(wa, wb)) / 4.0


def B6_HSM_5D(wa: WorkloadWindow, wb: WorkloadWindow) -> float:
    """B6: Full 5-D HSM (proposed). Uses DEFAULT_WEIGHTS."""
    d = hsm_score(wa, wb, DEFAULT_WEIGHTS)
    return d['HSM']


# ─── Registry ────────────────────────────────────────────────────────────────

BASELINES = {
    'B0  S_V (volume-ratio)': (B0_volume_ratio,  '1/5'),
    'B1  Cosine-QT':          (B1_cosine_qt,     '1/5'),
    'B2  Jaccard-Schema':     (B2_jaccard_schema, '1/5'),
    'B3  HSM-2D':             (B3_HSM_2D,        '2/5'),
    'B4  HSM-3D':             (B4_HSM_3D,        '3/5'),
    'B5  HSM-4D (no S_P)':   (B5_HSM_4D,        '4/5'),
    'B6  HSM-5D (proposed)':  (B6_HSM_5D,        '5/5'),
}
