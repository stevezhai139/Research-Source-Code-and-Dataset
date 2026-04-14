# HSM_gated — Hierarchical Similarity Measurement for Gated Index Management

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL-16-336791.svg)](https://www.postgresql.org/)

Replication package for the **HSM** framework — a hierarchical, multi-resolution
workload-similarity measure that gates index advisor invocations to suppress
false-alarm rebuilds. This sub-folder contains everything needed to reproduce
the empirical results in the paper.

---

## At a Glance

| Item                           | Value                                                                |
| ------------------------------ | -------------------------------------------------------------------- |
| Optimal HSM window             | $N^{\star} \in [50, 100]$ (closed-form, Theorem 6)                   |
| Detector quality (TPC-H)       | $\hat J = 1.000$ (Hoeffding 95% LB $\geq 0.77$)                      |
| Cross-scale invariance         | $N^{\star}$ unchanged across $15\times$ change in $N$ (SF 0.2 → 3.0) |
| Throughput gain over advisor   | See `paper/HSM_main.pdf` Section VI                                  |
| Total replication time         | $\sim$8 hours on a 16-core machine with 32 GB RAM                    |

---

## Repository Layout

```
HSM_gated/
├── README.md               ← you are here
├── LICENSE                 BSD-3-Clause
├── CITATION.cff            machine-readable citation metadata
├── .env.example            template for local credentials (copy → .env)
├── .gitignore
├── paper/
│   ├── HSM_main.pdf            ← main article (12 pages)
│   ├── HSM_supplementary.pdf   ← proofs + extra experiments (9 pages)
│   ├── main_article.tex
│   ├── supplementary.tex
│   └── figures/                13 PNG figures, ≥300 dpi
├── code/
│   ├── postgresql.conf         tuned config for native PostgreSQL
│   ├── requirements.txt        Python dependencies
│   ├── setup/                  one-shot build & load scripts
│   │   ├── 00_compile_and_generate.sh    TPC-H dbgen build + .tbl generation
│   │   ├── 01_create_tables.sql          TPC-H schema
│   │   ├── 02_load_data.sh               COPY .tbl into PostgreSQL
│   │   └── 03_create_base_indexes.sql    FK / join indexes
│   ├── docker/                 8 GB-capped PostgreSQL container
│   │   ├── docker-compose.yml
│   │   ├── postgresql_docker.conf
│   │   ├── load_data.sh                  pg_dump → restore (native → docker)
│   │   └── load_imdb.sh                  IMDB CSV → JOB benchmark
│   └── experiments/            Python entry-points for every experiment
│       ├── hsm_similarity.py             core HSM algorithm
│       ├── workload_generator.py         synthetic + TPC-H trace builder
│       ├── experiment_runner.py          main throughput study
│       ├── hsm_oltp_validation.py        OLTP detector validation
│       ├── hsm_burst_validation.py       burst-traffic detector
│       ├── hsm_burst_v2_validation.py    isolation-controlled rerun
│       ├── hsm_burst_v3_validation.py    randomized-block design
│       ├── hsm_job_validation.py         JOB / IMDB benchmark
│       ├── hsm_job_complexity_validation.py
│       └── hsm_sdss_validation.py        SDSS public log validation
├── data/
│   └── README.md           download instructions for TPC-H, JOB, SDSS
└── docs/
    └── REPRODUCE.md        step-by-step replication guide
```

---

## Quick Start

> Full step-by-step replication is in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).
> The summary below assumes Docker, PostgreSQL 16, and Python 3.9+ are installed.

```bash
# 1. Configure credentials
cp .env.example .env
$EDITOR .env                          # set HSM_DOCKER_PASSWORD etc.
source .env

# 2. Bring up the memory-capped PostgreSQL container
cd code/docker && docker compose up -d && cd ../..

# 3. Build TPC-H dbgen + generate scale-factor data
bash code/setup/00_compile_and_generate.sh

# 4. Load into Docker container
bash code/docker/load_data.sh

# 5. Install Python dependencies
python -m pip install -r code/requirements.txt

# 6. Run the headline experiment (4 SF × 4 conditions × 10 reps)
python code/experiments/experiment_runner.py --port 5433
```

---

## Reproducing Specific Figures / Tables

| Paper artefact          | Script                                          |
| ----------------------- | ----------------------------------------------- |
| Throughput (Fig. 7)     | `experiment_runner.py`                          |
| OLTP detector (Sec. VI) | `hsm_oltp_validation.py --execute`              |
| Burst detector          | `hsm_burst_v3_validation.py`                    |
| JOB / IMDB benchmark    | `hsm_job_validation.py --execute`               |
| SDSS public-log study   | `hsm_sdss_validation.py`                        |
| Window-sweep (Tbl. S-1) | `experiment_runner.py --sweep-window`           |

---

## Datasets

All raw datasets are public; download instructions are in
[`data/README.md`](data/README.md). We do **not** redistribute them due to
licensing and size (~20 GB at SF=10).

* **TPC-H** v3.0.1 — generated locally via `dbgen` (script provided).
* **JOB** (Join Order Benchmark) — IMDB CSV dump from
  [University of Mannheim mirror](https://event.cwi.nl/da/job/imdb.tgz).
* **SDSS** — sample of the public SQL workload log
  (see `data/README.md` for the exact subset and date range).

---

## Hardware Requirements

* **Disk:** ≥ 50 GB free for SF=10 data + Docker volume.
* **RAM:** ≥ 16 GB host (Docker container is hard-capped at 8 GB to force
  realistic disk I/O for SF=10).
* **CPU:** any x86-64 or Apple Silicon. Single-threaded experiments;
  no GPU required.

---

## Citation

If this code or paper is useful, please cite:

```bibtex
@article{hsm2026,
  title  = {HSM: Hierarchical Similarity Measurement for Gated Index Management},
  author = {Reungsinkonkarn, Arun and ...},
  year   = {2026},
  note   = {Manuscript under review},
}
```

The same metadata is also in [`CITATION.cff`](CITATION.cff) (machine-readable).

---

## License

BSD 3-Clause — see [`LICENSE`](LICENSE). Datasets retain their original licenses.

---

## Contact / Issues

Please open an issue on this repository for replication problems or questions
about the code. For paper-related questions, contact the corresponding author
listed in the article.
