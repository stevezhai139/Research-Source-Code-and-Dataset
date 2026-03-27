"""
gating_efficiency.py
====================
Theorem 9 Proof-of-Concept: HSM as a Gating Layer over SOTA Index Systems.

Mathematical framework:
  Let A = any index management system (BALANCE, MFIX, Indexer++) with:
    T_A(N)    = per-invocation cost = Ω(N log N)
    T_HSM     = Θ(N_pts)  [Theorem 4, fixed window size]
    p_stable  = fraction of windows where workload is STABLE (no rebuild needed)
    θ*        = optimal HSM decision threshold

  Without HSM gating:
    Cost_naive(N, K) = K × T_A(N)               [invoke A on every window]

  With HSM gating:
    Cost_gated(N, K) = K × T_HSM + (1-p_stable) × K × T_A(N)

  Savings:
    S(N, K) = Cost_naive - Cost_gated
            = p_stable × K × T_A(N) - K × T_HSM
            = K × [p_stable × Ω(N log N) - Θ(N_pts)]

  As N → ∞ (N_pts fixed):
    lim_{N→∞} S(N,K) / T_A(N) = p_stable > 0     (Theorem 9 claim)

This file generates Table: Theoretical Savings Under HSM Gating
for different N, p_stable values.
"""

import numpy as np
import math

# ─── Cost model parameters ─────────────────────────────────────────────────
# From Paper 3A Table 2 calibration (TPC-H SF=0.2)
C_HSM_PER_WINDOW_MS = 1.09      # measured (Table 5): HSM overhead
C_CREATE_PER_N      = 2.1e-5    # ms per row for index creation (approx B-tree)
C_SCAN_PER_N        = 1.4e-6    # ms per row for sequential scan

def T_A(N: int, a: float = 2.1e-5, g: float = 1e-6) -> float:
    """Cost of one index advisor invocation: Θ(N log N) ms."""
    return a * N * math.log2(N) + g * N

def T_HSM(N_pts: int = 19, c: float = 0.0574) -> float:
    """Cost of one HSM computation: Θ(N_pts) ms [Theorem 4]."""
    return c * N_pts   # ≈ 1.09ms at N_pts=19

def compute_savings_table(
    N_values   = [10_000, 100_000, 500_000, 1_000_000, 5_000_000],
    p_values   = [0.50, 0.65, 0.80],
    K          = 100,       # number of windows evaluated
    N_pts      = 19,
) -> list:
    """Return rows for the savings comparison table."""
    rows = []
    for N in N_values:
        t_a   = T_A(N)
        t_hsm = T_HSM(N_pts)
        cost_naive = K * t_a
        for p in p_values:
            cost_gated  = K * t_hsm + (1 - p) * K * t_a
            savings_abs = cost_naive - cost_gated          # ms
            savings_pct = 100.0 * savings_abs / cost_naive
            ratio       = cost_naive / cost_gated
            rows.append({
                'N':            N,
                'p_stable':     p,
                'T_A_ms':       round(t_a, 2),
                'T_HSM_ms':     round(t_hsm, 4),
                'Cost_naive_s': round(cost_naive / 1000, 1),
                'Cost_gated_s': round(cost_gated / 1000, 1),
                'Savings_pct':  round(savings_pct, 1),
                'Speedup':      round(ratio, 1),
            })
    return rows


def compute_break_even_N(p_stable: float, N_pts: int = 19) -> float:
    """
    Minimum N at which HSM gating becomes beneficial.
    Solve: p_stable × T_A(N) = T_HSM(N_pts)
    i.e., p_stable × a × N × log₂N = c × N_pts
    """
    target = T_HSM(N_pts) / p_stable
    # Binary search
    lo, hi = 1, 10_000_000
    for _ in range(100):
        mid = (lo + hi) // 2
        if T_A(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def print_savings_table(rows: list) -> None:
    print(f"\n{'N':>10} {'p_stable':>9} {'T_A(ms)':>10} {'Naive(s)':>10} "
          f"{'Gated(s)':>10} {'Savings%':>10} {'Speedup':>8}")
    print('-' * 75)
    prev_N = None
    for r in rows:
        if r['N'] != prev_N and prev_N is not None:
            print()
        print(f"{r['N']:>10,} {r['p_stable']:>9.0%} {r['T_A_ms']:>10.2f} "
              f"{r['Cost_naive_s']:>10.1f} {r['Cost_gated_s']:>10.1f} "
              f"{r['Savings_pct']:>9.1f}% {r['Speedup']:>7.1f}×")
        prev_N = r['N']


if __name__ == '__main__':
    rows = compute_savings_table()
    print("=== HSM Gating Efficiency: Theoretical Savings (K=100 windows) ===")
    print_savings_table(rows)

    print("\n=== Break-Even Database Size ===")
    for p in [0.50, 0.65, 0.80]:
        N_be = compute_break_even_N(p)
        print(f"  p_stable={p:.0%}: HSM gating beneficial for N ≥ {N_be:,} rows")

    # Key asymptotic claim for Theorem 9
    print("\n=== Asymptotic Ratio lim_{N→∞} S(N)/T_A(N) ===")
    for p in [0.50, 0.65, 0.80]:
        limit = p   # = p_stable (the constant)
        print(f"  p_stable={p:.0%}:  lim = {limit:.2f}  "
              f"(savings converge to {100*p:.0f}% of index advisor cost)")
