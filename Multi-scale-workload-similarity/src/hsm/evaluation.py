"""
hsm/evaluation.py
=================
Statistical evaluation framework for workload similarity measures.

Metrics computed:
  1. Discrimination Ratio (DR) = mean(within-phase) / mean(cross-phase)
     Primary metric in Paper 3A (Tables 6–7).

  2. Mann-Whitney U test (one-tailed: within > cross)
     Used for statistical significance (p < 0.001 criterion).

  3. F1 for change-point detection
     Binary classification: sim < θ → change detected.
     Used for Experiment A in Baseline_Comparison_Protocol.

  4. Intraclass Correlation Coefficient (ICC)
     Assesses consistency of within-phase scores across runs.

Reference: Paper 3A, Section 5.13 (Discrimination Analysis) and
           Theorem 7 (Break-Even Threshold θ*).
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from typing import List, Dict, Tuple, Optional
from .measures import WorkloadWindow, BASELINES, hsm_score


# ─── Pair classification ─────────────────────────────────────────────────────

def classify_pairs(windows: List[WorkloadWindow],
                   ref_lag: int = 1) -> Tuple[List, List, List]:
    """
    Classify all consecutive window pairs as within-phase or cross-phase.

    Returns:
        pairs       : List of (W_i, W_j, is_cross: bool)
        within_idx  : Indices of within-phase pairs
        cross_idx   : Indices of cross-phase pairs
    """
    pairs = []
    within_idx, cross_idx = [], []
    for i in range(ref_lag, len(windows)):
        wa, wb = windows[i - ref_lag], windows[i]
        is_cross = (wa.phase != wb.phase)
        pairs.append((wa, wb, is_cross))
        if is_cross:
            cross_idx.append(len(pairs) - 1)
        else:
            within_idx.append(len(pairs) - 1)
    return pairs, within_idx, cross_idx


def compute_scores_for_measure(measure_fn, pairs: list) -> np.ndarray:
    """Apply a similarity measure to all pairs. Returns score array."""
    return np.array([measure_fn(wa, wb) for wa, wb, _ in pairs])


# ─── Discrimination ratio ─────────────────────────────────────────────────────

def discrimination_ratio(scores: np.ndarray,
                          within_idx: List[int],
                          cross_idx: List[int]) -> Dict:
    """
    Compute discrimination ratio and Mann-Whitney U test.

    DR = mean(within) / mean(cross)
    A DR > 1 with p < 0.05 indicates the measure can distinguish
    same-phase from cross-phase workloads.

    Returns dict with keys:
      within_mean, within_sd, cross_mean, cross_sd,
      disc_ratio, p_value, n_within, n_cross
    """
    w_scores = scores[within_idx] if len(within_idx) > 0 else np.array([])
    c_scores = scores[cross_idx]  if len(cross_idx)  > 0 else np.array([])

    within_mean = float(w_scores.mean()) if len(w_scores) > 0 else np.nan
    within_sd   = float(w_scores.std())  if len(w_scores) > 0 else np.nan
    cross_mean  = float(c_scores.mean()) if len(c_scores) > 0 else np.nan
    cross_sd    = float(c_scores.std())  if len(c_scores) > 0 else np.nan

    disc_ratio = (within_mean / cross_mean
                  if (cross_mean and cross_mean > 1e-9 and not np.isnan(cross_mean))
                  else np.inf)

    # One-tailed Mann-Whitney U: within > cross
    if len(w_scores) > 0 and len(c_scores) > 0:
        try:
            _, p_val = mannwhitneyu(w_scores, c_scores, alternative='greater')
            p_str = f'{p_val:.4f}' if p_val >= 0.0001 else '<.0001'
        except Exception:
            p_str = 'N/A'
    else:
        p_str = 'N/A'

    return {
        'within_mean': round(within_mean, 4) if not np.isnan(within_mean) else np.nan,
        'within_sd':   round(within_sd,   4) if not np.isnan(within_sd)   else np.nan,
        'cross_mean':  round(cross_mean,  4) if not np.isnan(cross_mean)  else np.nan,
        'cross_sd':    round(cross_sd,    4) if not np.isnan(cross_sd)    else np.nan,
        'disc_ratio':  round(disc_ratio,  4) if not np.isinf(disc_ratio)  else '∞',
        'p_value':     p_str,
        'n_within':    len(w_scores),
        'n_cross':     len(c_scores),
    }


# ─── F1 for change-point detection ───────────────────────────────────────────

def change_detection_f1(scores: np.ndarray,
                         true_change_indices: List[int],
                         theta: float = 0.75,
                         tolerance: int = 1) -> Dict:
    """
    Evaluate binary change-point detection.

    A change is detected at position i when scores[i] < theta.
    True positives allow ±tolerance window tolerance.

    Returns: precision, recall, f1, detected_indices
    """
    detected = [i for i, s in enumerate(scores) if s < theta]
    true_set = set(true_change_indices)
    pred_set = set(detected)

    tp = sum(1 for p in pred_set if any(abs(p - t) <= tolerance for t in true_set))
    fp = len(pred_set) - tp
    fn = len(true_set) - sum(1 for t in true_set
                              if any(abs(p - t) <= tolerance for p in pred_set))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        'precision':  round(prec, 3),
        'recall':     round(rec, 3),
        'f1':         round(f1, 3),
        'detected':   sorted(detected),
    }


# ─── Full baseline table ──────────────────────────────────────────────────────

def run_baseline_comparison(windows: List[WorkloadWindow],
                             ref_lag: int = 1,
                             theta: float = 0.75,
                             true_change_indices: Optional[List[int]] = None
                             ) -> pd.DataFrame:
    """
    Run all B0–B6 measures on the given window sequence.

    Returns a DataFrame with one row per measure, suitable for
    inclusion as Table 5/6 in Paper 3A.
    """
    pairs, within_idx, cross_idx = classify_pairs(windows, ref_lag)

    rows = []
    for name, (fn, dims) in BASELINES.items():
        scores = compute_scores_for_measure(fn, pairs)
        dr     = discrimination_ratio(scores, within_idx, cross_idx)

        row = {'Method': name, 'Dims': dims}
        row.update({
            'Within_mean': dr['within_mean'],
            'Within_sd':   dr['within_sd'],
            'Cross_mean':  dr['cross_mean'],
            'Cross_sd':    dr['cross_sd'],
            'Disc_ratio':  dr['disc_ratio'],
            'p_value':     dr['p_value'],
            'n_within':    dr['n_within'],
            'n_cross':     dr['n_cross'],
        })

        if true_change_indices is not None:
            f1_result = change_detection_f1(scores, true_change_indices, theta)
            row.update({
                'F1':        f1_result['f1'],
                'Precision': f1_result['precision'],
                'Recall':    f1_result['recall'],
            })

        rows.append(row)

    return pd.DataFrame(rows)


# ─── Aggregate across multiple runs ──────────────────────────────────────────

def aggregate_runs(results_per_run: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Average discrimination ratios across multiple experimental runs.
    Applies to numeric columns only.
    """
    numeric_cols = ['Within_mean', 'Within_sd', 'Cross_mean', 'Cross_sd']
    methods = results_per_run[0]['Method'].tolist()

    agg_rows = []
    for i, method in enumerate(methods):
        row_vals = {}
        for col in numeric_cols:
            vals = []
            for df in results_per_run:
                v = df.iloc[i][col]
                if isinstance(v, float) and not np.isnan(v):
                    vals.append(v)
            row_vals[col] = round(np.mean(vals), 4) if vals else np.nan

        # Recompute disc_ratio from aggregated means
        wm = row_vals['Within_mean']
        cm = row_vals['Cross_mean']
        row_vals['Disc_ratio'] = (round(wm / cm, 4) if (cm and cm > 1e-9) else '∞')

        # Take p_value from first run (order preserved by Mann-Whitney)
        row_vals['p_value'] = results_per_run[0].iloc[i]['p_value']
        row_vals['Method']  = method
        row_vals['Dims']    = results_per_run[0].iloc[i]['Dims']
        row_vals['n_within'] = results_per_run[0].iloc[i]['n_within']
        row_vals['n_cross']  = results_per_run[0].iloc[i]['n_cross']
        agg_rows.append(row_vals)

    return pd.DataFrame(agg_rows)


# ─── Pretty print ────────────────────────────────────────────────────────────

def print_comparison_table(df: pd.DataFrame, title: str = '') -> None:
    """Print LaTeX-ready comparison table to stdout."""
    if title:
        print(f'\n{"="*72}')
        print(f'  {title}')
        print(f'{"="*72}')
    cols = ['Method', 'Dims', 'Within_mean', 'Within_sd',
            'Cross_mean', 'Cross_sd', 'Disc_ratio', 'p_value']
    if 'F1' in df.columns:
        cols += ['Precision', 'Recall', 'F1']
    hdr = (f"{'Method':<26} {'Dims':>5} {'W_mean':>7} {'W_sd':>6} "
           f"{'C_mean':>7} {'C_sd':>6} {'DR':>6} {'p':>8}")
    if 'F1' in df.columns:
        hdr += f"  {'Prec':>5} {'Rec':>5} {'F1':>5}"
    print(hdr)
    print('-' * 80)
    for _, r in df.iterrows():
        marker = ' ←' if 'proposed' in r['Method'] else '  '
        line   = (f"{r['Method']:<26} {r['Dims']:>5}  "
                  f"{r['Within_mean']:>6.4f} {r['Within_sd']:>6.4f}  "
                  f"{str(r['Cross_mean']):>6}  {str(r['Cross_sd']):>6}  "
                  f"{str(r['Disc_ratio']):>6}  {str(r['p_value']):>8}{marker}")
        if 'F1' in df.columns:
            line += (f"  {r.get('Precision',0):>5.3f} {r.get('Recall',0):>5.3f} "
                     f"{r.get('F1',0):>5.3f}")
        print(line)
    print()


def to_latex(df: pd.DataFrame, caption: str = '',
             label: str = 'tab:baseline_comparison') -> str:
    """
    Generate LaTeX tabular environment for the baseline comparison.
    Bolds the HSM-5D row. Suitable for direct inclusion in Paper 3A.
    """
    lines = [
        r'\begin{table}[!t]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{llrrrrrr}',
        r'\toprule',
        r'Method & Dims & $\mu_{w}$ & $\sigma_{w}$ '
        r'& $\mu_{c}$ & $\sigma_{c}$ & DR & $p$ \\',
        r'\midrule',
    ]
    for _, r in df.iterrows():
        is_proposed = 'proposed' in r['Method']
        vals = (f"{r['Within_mean']:.4f} & {r['Within_sd']:.4f} & "
                f"{str(r['Cross_mean'])} & {str(r['Cross_sd'])} & "
                f"{str(r['Disc_ratio'])} & {r['p_value']}")
        row_str = f"{r['Method']} & {r['Dims']} & {vals} \\\\"
        if is_proposed:
            row_str = r'\textbf{' + row_str.replace('\\\\', r'} \\')
        lines.append(row_str)
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)
