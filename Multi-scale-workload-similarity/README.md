# HSM: Hierarchical Similarity Measurement for Proactive Database Index Tuning

Replication package for:

> Arun Reungsilpkolkarn. "Multi-Scale Workload Similarity Measurement for Proactive Database Index Tuning using Hierarchical Similarity Measurement (HSM)." *Information Systems*, 2026. (under review)

---

## Repository Structure

```
├── src/hsm/                          # Core HSM implementation
│   ├── measures.py                   # Five-dimensional similarity (S_R, S_V, S_P, S_A, S_C)
│   ├── windowing.py                  # Query window construction + DWT band computation
│   └── evaluation.py                # Evaluation & ICC utilities
│
├── experiments/                      # Experiment scripts
│   ├── 01_baseline_tpch.py          # TPC-H baseline comparison (Tables 5–6)
│   ├── 02_baseline_sdss.py          # SDSS baseline comparison (Table 7)
│   ├── 03_scale_tpch.py             # Scalability: TPC-H SF 0.2–10 (Table 8, Fig. 6)
│   ├── 04_sdss_realdata.py          # SDSS SkyServer real-data validation (Table 9, Fig. 9)
│   ├── run_experiment.py            # Main TPC-H experiment runner (requires PostgreSQL)
│   └── run_scale_experiment.py      # Scale experiment runner (requires PostgreSQL)
│
├── baseline/                         # Competing methods implementation
│   ├── baseline_measures.py         # Cosine, Jaccard, DTW, histogram baselines
│   ├── gating_efficiency.py         # Theorem 9 gating efficiency analysis
│   ├── simulate_tpch_windows.py     # TPC-H workload window simulator
│   └── Baseline_Comparison_Protocol_Paper3A.md
│
├── data/
│   └── tpch_sf0.2/README.md         # TPC-H data generation instructions
│
├── results/                          # Pre-generated experimental results (CSV)
│   ├── tpch/
│   │   ├── run_01/ … run_05/        # Five independent TPC-H runs (raw CSV)
│   │   └── summary/                 # Aggregated summary tables
│   ├── sdss/                        # SDSS SkyServer results
│   └── scale/                       # Scalability results (SF 0.05, 0.2, 1, 3, 10)
│
└── figures/                          # Reproduced paper figures (PNG)
    ├── fig5_hsm_timeline.png
    ├── fig6_timing_loglog.png
    ├── fig7_adaptation_trigger.png
    ├── fig8_radar.png
    └── fig9_tpch_vs_sdss.png
```

---

## Requirements

### Python dependencies

```bash
pip install -r requirements.txt
```

### PostgreSQL (required for live experiments)

`run_scale_experiment.py` and `run_experiment.py` connect to a **live PostgreSQL server**
to perform real `CREATE INDEX` / `DROP INDEX` operations and measure T_A (actual index
rebuild time). Pre-generated results are already provided in `results/` if you only
want to verify the numbers without re-running.

**Install PostgreSQL:**

- **macOS**: `brew install postgresql && brew services start postgresql`
- **Ubuntu/Debian**: `sudo apt install postgresql && sudo systemctl start postgresql`
- **Windows**: Download from https://www.postgresql.org/download/windows/

**Configure credentials:**

Edit the `PG` block near the top of `run_scale_experiment.py`:

```python
PG = dict(
    host     = "localhost",
    port     = 5432,
    user     = "your_username",   # your OS username (macOS) or "postgres"
    password = "",                # leave empty for peer auth (macOS/Linux)
    dbname   = "postgres",
)
```

**Verify connection:**

```bash
psql -U your_username -c "SELECT version();"
```

---

## Reproducing Results

> **Note:** All results are pre-generated in `results/`. The steps below are for
> full re-reproduction from scratch.

### Option A — Verify results only (no PostgreSQL needed)

All CSV files in `results/` can be inspected directly. Summary tables correspond
to paper tables as follows:

| File | Paper |
|------|-------|
| `results/tpch/summary/summary_hsm.csv` | Table 5 (HSM scores) |
| `results/tpch/summary/summary_timing.csv` | Table 8 (T_A timing) |
| `results/tpch/tpch_baseline_summary_new.csv` | Table 6 (baseline comparison) |
| `results/sdss/sdss_baseline_summary_new.csv` | Table 7 (SDSS baseline) |
| `results/scale/scale_results_summary.csv` | Table 8 (scale summary) |

### Option B — Re-run all experiments (requires PostgreSQL)

**Step 1 — Generate TPC-H data:**
See `data/tpch_sf0.2/README.md` for TPC-H dbgen instructions.

**Step 2 — Run TPC-H baseline (Tables 5–6):**
```bash
python experiments/01_baseline_tpch.py
```

**Step 3 — Run SDSS baseline (Table 7):**
```bash
python experiments/02_baseline_sdss.py
```

**Step 4 — Run scalability experiment (Table 8, Figure 6):**
```bash
# Single scale factor (recommended):
python experiments/run_scale_experiment.py --scale 1.0 --runs 5

# All scale factors (warning: SF=10 takes ~21 hours):
python experiments/run_scale_experiment.py --scale 0.2 1.0 3.0 10.0 --runs 5
```

Estimated runtimes per run (M4 MacBook Pro):

| Scale Factor | Time/run | ×5 runs |
|---|---|---|
| SF=0.05 | ~2 min | ~10 min (smoke test) |
| SF=0.2 | ~8 min | ~40 min |
| SF=1.0 | ~25 min | ~2 hr |
| SF=3.0 | ~70 min | ~6 hr |
| SF=10.0 | ~250 min | ~21 hr |

**Step 5 — Run SDSS real-data validation (Table 9, Figure 9):**
```bash
python experiments/04_sdss_realdata.py
```

---

## Data Sources

**TPC-H**: Generate using the official TPC-H benchmark toolkit.
See `data/tpch_sf0.2/README.md` for generation instructions.
Official toolkit: http://www.tpc.org/tpch/

**SDSS SkyServer**: Publicly available at https://skyserver.sdss.org/
Query logs used in this paper are included in `results/sdss/`.

---

## Contact

Arun Reungsilpkolkarn
Department of Computer Science and Information Technology, Bangkok University, Pathum Thani, Thailand
arun.r@bu.ac.th
