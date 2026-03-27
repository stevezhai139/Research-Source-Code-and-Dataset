"""
run_experiments.py
==================
Main experiment runner. Produces:
  Table A : Baseline comparison on workload change detection
  Table B : HSM Gating Efficiency (theoretical savings)
"""

import sys
sys.path.insert(0, '/tmp/baseline_experiment')

from simulate_tpch_windows import generate_windows
from baseline_measures import evaluate_change_detection, print_comparison_table
from gating_efficiency import compute_savings_table, print_savings_table

# ─── Experiment A: Baseline Comparison ─────────────────────────────────────
print("=" * 65)
print("EXPERIMENT A: Baseline Similarity Measures vs HSM-5D")
print("Dataset: TPC-H SF=0.2  |  31 windows  |  3 change points")
print("=" * 65)

windows = generate_windows(31)
TRUE_CHANGE_POINTS = [7, 15, 23]   # from paper

# Evaluate at multiple thresholds, pick θ=0.75 (from paper)
results = evaluate_change_detection(windows, TRUE_CHANGE_POINTS, theta=0.75)
print_comparison_table(results)

print(f"\nGround-truth change points: {TRUE_CHANGE_POINTS}")
print("Metric: Precision / Recall / F1 with ±1 window tolerance\n")

# ─── Experiment B: Gating Efficiency ───────────────────────────────────────
print("=" * 65)
print("EXPERIMENT B: HSM Gating Savings over SOTA Index Systems")
print("Model: T_A(N) = Ω(N log N)  [BALANCE/MFIX/Indexer++ class]")
print("=" * 65)

rows = compute_savings_table(K=100)
print_savings_table(rows)

print("\nNote: T_A models ANY system requiring index rebuild (Θ(N log N)).")
print("Results apply generically to BALANCE [Wang2024], MFIX [Chang2024],")
print("Indexer++ [Sharma2022] — all perform physical index operations on N rows.")
