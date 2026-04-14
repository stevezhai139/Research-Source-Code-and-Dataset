# Step-by-Step Reproduction Guide

This document walks you through reproducing every empirical result in
`paper/HSM_main.pdf`. Estimated total time on a 16-core / 32 GB machine:
**8 hours** (most of it spent on the SF=10 throughput study).

---

## 0. Prerequisites

| Component        | Tested version       | Install hint                                  |
| ---------------- | -------------------- | --------------------------------------------- |
| Operating system | Ubuntu 22.04 / macOS 13+ | —                                         |
| PostgreSQL       | 16.x                 | `apt install postgresql-16` / `brew install postgresql@16` |
| Docker           | 24.0+                | https://docs.docker.com/get-docker/           |
| Docker Compose   | v2 (built-in)        | bundled with Docker Desktop                   |
| Python           | 3.9 – 3.12           | system package or `pyenv`                     |
| C compiler       | gcc 11+ / Apple Clang| `apt install build-essential` / `xcode-select --install` |
| Disk space       | 50 GB free           | for SF=10 + Docker volume                     |
| RAM              | ≥ 16 GB              | Docker container is hard-capped at 8 GB       |

---

## 1. Clone & Configure

```bash
git clone https://github.com/stevezhai139/Research-Source-Code-and-Dataset.git
cd Research-Source-Code-and-Dataset/HSM_gated

cp .env.example .env
$EDITOR .env                        # set passwords
source .env                         # export to current shell

python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
```

The `.env` file is git-ignored. Required variables:

| Variable                 | Default      | Purpose                                |
| ------------------------ | ------------ | -------------------------------------- |
| `HSM_DB_HOST`            | `localhost`  | native PostgreSQL host                 |
| `HSM_DB_PORT`            | `5432`       | native PostgreSQL port                 |
| `HSM_DB_USER`            | `postgres`   | native PostgreSQL user                 |
| `HSM_DB_PASSWORD`        | (empty)      | native PostgreSQL password             |
| `HSM_DOCKER_HOST`        | `localhost`  | Docker container host                  |
| `HSM_DOCKER_PORT`        | `5433`       | Docker exposed port                    |
| `HSM_DOCKER_USER`        | `postgres`   | container superuser                    |
| `HSM_DOCKER_PASSWORD`    | `postgres`   | container password (CHANGE for non-localhost!) |
| `HSM_IMDB_DB`            | `imdb`       | logical name of IMDB database          |

---

## 2. Generate TPC-H Data

This compiles `dbgen` from the official TPC-H 3.0.1 source and produces
`.tbl` files for SF ∈ {0.2, 1.0, 3.0, 10.0}. Total: ~25 GB on disk,
~30 minutes wall-clock on an SSD.

```bash
bash code/setup/00_compile_and_generate.sh
# → produces code/data/sf0.2/, sf1/, sf3/, sf10/
```

Generate a single SF only:

```bash
bash code/setup/00_compile_and_generate.sh 0.2
```

---

## 3. Bring Up Docker PostgreSQL

The container is hard-capped at 8 GB RAM with `mem_limit: 8g` so that
SF=10 (~16.7 GB) cannot fully reside in memory. This is **required** to
reproduce realistic disk-I/O behaviour reported in the paper.

```bash
cd code/docker
docker compose up -d                 # start
docker compose ps                    # verify "healthy"
cd ../..
```

Then load the SF data into the container:

```bash
bash code/docker/load_data.sh        # all 4 SF (sequential, ~2 hours)
bash code/docker/load_data.sh 0.2    # single SF
```

---

## 4. Run Experiments

### 4.1 Headline throughput study (paper Fig. 7, Table 4)

```bash
python code/experiments/experiment_runner.py --port 5433
# → results/sf{0.2,1,3,10}/raw_results.csv
# → results/summary.csv
```

Quick smoke test (single SF, 3 reps, ~5 min):

```bash
python code/experiments/experiment_runner.py --port 5433 --sf 0.2 --reps 3 --quick
```

### 4.2 OLTP detector validation (Sec. VI-B)

```bash
python code/experiments/hsm_oltp_validation.py --execute
# → results/oltp_validation/
```

### 4.3 Burst detector (Sec. VI-C)

`v3` uses the randomized-block design described in the paper. `v1`/`v2`
are kept for reviewers who want to inspect the methodological evolution.

```bash
python code/experiments/hsm_burst_v3_validation.py
```

### 4.4 JOB / IMDB benchmark (Sec. VI-D)

Requires IMDB CSV files in `code/data/imdb/`. Download instructions are
in [`../data/README.md`](../data/README.md).

```bash
bash code/docker/load_imdb.sh
python code/experiments/hsm_job_validation.py --execute
python code/experiments/hsm_job_complexity_validation.py
```

### 4.5 SDSS public-log validation (Sec. VI-E)

```bash
python code/experiments/hsm_sdss_validation.py
```

### 4.6 Window-sweep (Supplementary Table S-1, Corollary 1)

```bash
python code/experiments/experiment_runner.py --port 5433 --sweep-window
```

---

## 5. Cleanup

```bash
cd code/docker && docker compose down -v   # stop + delete volume
rm -rf code/data/sf*                       # remove generated TPC-H data
```

---

## Troubleshooting

| Symptom                                       | Likely cause / fix                                         |
| --------------------------------------------- | ---------------------------------------------------------- |
| `psycopg2.OperationalError: password authentication failed` | `.env` not sourced, or `HSM_DOCKER_PASSWORD` mismatched between the Python process and the container. |
| `dbgen: command not found`                    | Re-run `00_compile_and_generate.sh`; check Xcode CLI tools on macOS. |
| Container exits with `out of memory`          | Lower `mem_limit` only if you also rebuild base data at smaller SF. |
| SF=10 throughput appears too good to be true  | Verify `mem_limit: 8g` is enforced: `docker stats tpch_docker`. |
| `pg_isready` hangs                            | Container not healthy yet; `docker compose logs postgres`. |

For anything not listed here, open an issue on this repository with:

* OS + version
* `docker --version` and `docker compose version`
* Full traceback / log output
* Which step from this guide failed
