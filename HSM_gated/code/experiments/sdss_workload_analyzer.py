"""
sdss_workload_analyzer.py
=========================
Real-World Workload Validation using SDSS SkyServer Query Logs
HSM Throughput Experiments — Section 5 (Cross-Workload Validation)

Purpose:
  Demonstrates that HSM correctly detects workload drift in real-world
  astronomical SQL query logs, independent of the synthetic TPC-H benchmark.
  This serves as the "real-world robustness" evidence for HSM.

Input:
  SDSS SkyLog_Workload.csv downloaded from CasJobs (SdssWeblogs.SQLlogAll)
  Columns: statement, theTime, elapsed, busy, rows, dbname, error

Output:
  results/sdss/sdss_hsm_scores.csv      — HSM score per window
  results/sdss/sdss_drift_events.csv    — detected drift events
  results/sdss/fig_sdss_hsm_trace.pdf   — HSM score timeline
  results/sdss/fig_sdss_query_types.pdf — query pattern distribution

Usage:
  python experiments/sdss_workload_analyzer.py
  python experiments/sdss_workload_analyzer.py --csv path/to/SkyLog_Workload.csv
  python experiments/sdss_workload_analyzer.py --window 20 --theta 0.75
"""

import argparse
import csv
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from hsm_similarity import build_window, should_trigger_advisor, hsm_score, DEFAULT_THETA

# ─── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "sdss"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV    = BASE_DIR.parent / "SkyLog_Workload.csv"  # adjust if needed
WINDOW_SIZE    = 20    # queries per HSM window
THETA          = DEFAULT_THETA  # 0.75
MAX_ROWS       = 100_000        # cap for memory efficiency


# ─── SDSS Query Classifier ────────────────────────────────────────────────────

SDSS_QUERY_PATTERNS = {
    "PhotoObj":    r"\bPhotoObj\b|\bPhotoObjAll\b|\bPhotoTag\b",
    "SpecObj":     r"\bSpecObj\b|\bSpecObjAll\b|\bSpecLine\b",
    "Galaxy":      r"\bGalaxy\b|\bgalaxy\b",
    "Star":        r"\bStar\b|\bstar\b",
    "Quasar":      r"\bQuasar\b|\bQSO\b|\bqso\b",
    "Field":       r"\bField\b|\bRun\b|\bCamcol\b",
    "Coordinate":  r"\bra\b|\bdec\b|\bRA\b|\bDEC\b|fGetNearestObjEq",
    "Redshift":    r"\bredshift\b|\bz\b.*\bFROM\b",
    "Metadata":    r"\bDBObjects\b|\bDBColumns\b|\bHistory\b|sys\.",
}

def classify_query(sql: str) -> str:
    """Classify an SDSS query by its primary subject."""
    sql_lower = sql.lower() if sql else ""
    for category, pattern in SDSS_QUERY_PATTERNS.items():
        if re.search(pattern, sql, re.IGNORECASE):
            return category
    return "Other"


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_sdss_csv(csv_path: Path, max_rows: int = MAX_ROWS) -> list:
    """
    Load SDSS query log CSV.
    Returns list of dicts with keys: statement, theTime, elapsed, rows
    Sorted chronologically by theTime.
    """
    records = []
    print(f"Loading SDSS data from: {csv_path}")

    if not csv_path.exists():
        print(f"\nERROR: File not found: {csv_path}")
        print("Please specify the correct path with --csv flag:")
        print("  python experiments/sdss_workload_analyzer.py --csv ~/Downloads/SkyLog_Workload.csv")
        sys.exit(1)

    def safe_float(val, default=0.0):
        try:
            return float(val) if val else default
        except (ValueError, TypeError):
            return default

    def safe_int(val, default=0):
        try:
            return int(float(val)) if val else default
        except (ValueError, TypeError):
            return default

    with open(csv_path, encoding="utf-8", errors="replace") as f:
        # Use QUOTE_ALL to handle SQL statements with commas/newlines
        reader = csv.DictReader(f, quoting=csv.QUOTE_ALL)
        skipped = 0
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            stmt = row.get("statement", "").strip()
            # Skip empty, very short, or clearly malformed rows
            if not stmt or len(stmt) < 20:
                skipped += 1
                continue
            # Skip rows where statement looks like a number (column shift)
            if stmt.replace('.','').replace('-','').isdigit():
                skipped += 1
                continue
            records.append({
                "statement":  stmt,
                "theTime":    row.get("theTime", ""),
                "elapsed":    safe_float(row.get("elapsed", 0)),
                "rows":       safe_int(row.get("rows", 0)),
                "dbname":     row.get("dbname", ""),
                "query_type": classify_query(stmt),
            })
    if skipped > 0:
        print(f"  Skipped {skipped:,} malformed/empty rows")

    # Sort chronologically
    def parse_time(r):
        try:
            return datetime.fromisoformat(r["theTime"].replace("T", " "))
        except Exception:
            return datetime.min

    records.sort(key=parse_time)
    print(f"  Loaded {len(records):,} valid queries")
    return records


# ─── HSM Analysis on SDSS Workload ────────────────────────────────────────────

def analyze_workload(records: list, window_size: int, theta: float):
    """
    Slide a window over the SDSS query trace and compute HSM scores.

    Returns list of window stats dicts.
    """
    results = []
    prev_window = None
    n_windows = len(records) // window_size

    print(f"\nAnalyzing {n_windows} windows (window_size={window_size}, θ={theta})")
    print("-" * 50)

    for i in range(n_windows):
        start = i * window_size
        end   = start + window_size
        window_queries = [r["statement"] for r in records[start:end]]
        window_times   = [r["theTime"]   for r in records[start:end]]
        window_types   = [r["query_type"] for r in records[start:end]]
        avg_elapsed    = sum(r["elapsed"] for r in records[start:end]) / window_size

        curr_window = build_window(window_queries, window_id=i)

        trigger, score, dims = should_trigger_advisor(prev_window, curr_window, theta)

        # Dominant query type in this window
        type_counts    = Counter(window_types)
        dominant_type  = type_counts.most_common(1)[0][0]

        result = {
            "window_id":      i,
            "start_time":     window_times[0]  if window_times else "",
            "end_time":       window_times[-1] if window_times else "",
            "hsm_score":      round(score, 4),
            "drift_detected": trigger,
            "dominant_type":  dominant_type,
            "avg_elapsed_ms": round(avg_elapsed * 1000, 2),
            "S_R": round(dims.get("S_R", 0), 4),
            "S_V": round(dims.get("S_V", 0), 4),
            "S_T": round(dims.get("S_T", 0), 4),
            "S_A": round(dims.get("S_A", 0), 4),
            "S_P": round(dims.get("S_P", 0), 4),
        }
        results.append(result)
        prev_window = curr_window

        if i % 100 == 0:
            drift_str = "← DRIFT" if trigger else ""
            print(f"  Window {i:4d}: HSM={score:.3f}  type={dominant_type:<12} {drift_str}")

    # Summary
    drift_windows  = sum(1 for r in results if r["drift_detected"])
    stable_windows = len(results) - drift_windows
    drift_rate     = drift_windows / len(results) * 100 if results else 0

    print(f"\n{'='*50}")
    print(f"  Total windows:   {len(results)}")
    print(f"  Drift detected:  {drift_windows} ({drift_rate:.1f}%)")
    print(f"  Stable windows:  {stable_windows} ({100-drift_rate:.1f}%)")
    print(f"  Mean HSM score:  {sum(r['hsm_score'] for r in results)/len(results):.3f}")
    print(f"{'='*50}")

    return results


# ─── Save Results ─────────────────────────────────────────────────────────────

def save_results(results: list):
    """Save HSM analysis results to CSV."""
    scores_path = RESULTS_DIR / "sdss_hsm_scores.csv"
    drift_path  = RESULTS_DIR / "sdss_drift_events.csv"

    # All window scores
    with open(scores_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {scores_path}")

    # Drift events only
    drifts = [r for r in results if r["drift_detected"]]
    if drifts:
        with open(drift_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=drifts[0].keys())
            writer.writeheader()
            writer.writerows(drifts)
        print(f"Saved: {drift_path} ({len(drifts)} drift events)")


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_results(results: list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib not installed — skipping plots")
        print("  Run: pip install matplotlib numpy")
        return

    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                          "savefig.dpi": 300, "savefig.bbox": "tight"})

    windows  = [r["window_id"]  for r in results]
    scores   = [r["hsm_score"]  for r in results]
    drifts   = [r["window_id"]  for r in results if r["drift_detected"]]
    types    = [r["dominant_type"] for r in results]

    # ── Figure 1: HSM Score Trace ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(windows, scores, color="#4472C4", linewidth=0.8,
            alpha=0.9, label="HSM Score", zorder=3)

    # Shade drift regions
    for w in drifts:
        ax.axvspan(w - 0.5, w + 0.5, color="#FF6B00", alpha=0.25, zorder=1)

    ax.axhline(y=THETA, color="red", linestyle="--", linewidth=1.2,
               label=f"θ = {THETA}", zorder=2)

    ax.fill_between(windows, scores, THETA,
                    where=[s < THETA for s in scores],
                    color="#FF6B00", alpha=0.15, label="Drift zone")

    ax.set_xlabel("Window Index (each window = 20 queries)")
    ax.set_ylabel("HSM Similarity Score")
    ax.set_title(
        f"HSM Workload Drift Detection on SDSS SkyServer Query Logs\n"
        f"(n={len(results)} windows, {len(drifts)} drift events detected, θ={THETA})"
    )
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    for fmt in ("pdf", "png"):
        p = RESULTS_DIR / f"fig_sdss_hsm_trace.{fmt}"
        fig.savefig(p)
    plt.close(fig)
    print(f"  Saved: fig_sdss_hsm_trace.pdf/png")

    # ── Figure 2: Query Type Distribution ────────────────────────────────────
    type_counts = Counter(types)
    labels = list(type_counts.keys())
    values = list(type_counts.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                f"{v:,}", va="center", fontsize=8)

    ax.set_xlabel("Number of Windows")
    ax.set_title("SDSS Query Type Distribution across Workload Windows")
    ax.set_xlim(0, max(values) * 1.15)

    for fmt in ("pdf", "png"):
        p = RESULTS_DIR / f"fig_sdss_query_types.{fmt}"
        fig.savefig(p)
    plt.close(fig)
    print(f"  Saved: fig_sdss_query_types.pdf/png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SDSS Real-World Workload Validation — HSM"
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Path to SkyLog_Workload.csv (default: auto-detect)"
    )
    parser.add_argument(
        "--window", type=int, default=WINDOW_SIZE,
        help=f"Window size in queries (default: {WINDOW_SIZE})"
    )
    parser.add_argument(
        "--theta", type=float, default=THETA,
        help=f"HSM drift threshold (default: {THETA})"
    )
    parser.add_argument(
        "--max-rows", type=int, default=MAX_ROWS,
        help=f"Max rows to process (default: {MAX_ROWS:,})"
    )
    args = parser.parse_args()

    # ── Find CSV file ──────────────────────────────────────────────────────────
    csv_path = args.csv
    if csv_path is None:
        # Auto-detect common locations
        candidates = [
            BASE_DIR.parent / "SkyLog_Workload.csv",
            Path.home() / "Downloads" / "SkyLog_Workload.csv",
            Path.home() / "Desktop" / "SkyLog_Workload.csv",
            BASE_DIR / "data" / "SkyLog_Workload.csv",
        ]
        for c in candidates:
            if c.exists():
                csv_path = c
                print(f"  Auto-detected CSV: {csv_path}")
                break
        if csv_path is None:
            print("ERROR: SkyLog_Workload.csv not found.")
            print("Specify path with: --csv /path/to/SkyLog_Workload.csv")
            sys.exit(1)

    # ── Run analysis ───────────────────────────────────────────────────────────
    print("=" * 55)
    print("  HSM — SDSS Real-World Workload Validation")
    print(f"  Window size : {args.window} queries")
    print(f"  Theta       : {args.theta}")
    print(f"  Max rows    : {args.max_rows:,}")
    print("=" * 55)

    records = load_sdss_csv(csv_path, max_rows=args.max_rows)
    results = analyze_workload(records, args.window, args.theta)
    save_results(results)
    plot_results(results)

    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("Next: run analysis/compute_statistics.py for full summary")


if __name__ == "__main__":
    main()
