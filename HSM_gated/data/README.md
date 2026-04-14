# Data

This folder contains **processed** result CSVs only. Raw database traces are omitted because their combined size is ~20 GB; they are deterministically regenerated from the `code/setup/` scripts.

## Contents

```
data/
├── README.md                        ← you are here
└── results/
    ├── summary_stats.csv            ← DR, CI, p, ICC, r per workload (Tables II–V)
    ├── effect_sizes.csv             ← per-dimension effect sizes (dim-utility table)
    ├── table_throughput.tex         ← wall-QPS table (A13)
    ├── sf0.2/                       ← TPC-H SF=0.2 per-run + summary
    ├── sf1.0/                       ← TPC-H SF=1.0
    ├── sf3.0/                       ← TPC-H SF=3.0
    ├── sdss/                        ← SDSS SkyServer (A8)
    ├── job_validation/              ← JOB/IMDB (A8)
    ├── oltp_validation/             ← pgbench TPC-B (A8)
    ├── burst_validation/            ← burst workload (A8)
    ├── burst_v2_validation/
    └── burst_v3_validation/
```

## Regenerating the raw data

### TPC-H (SF 0.2 / 1.0 / 3.0)

```bash
cd code/setup
bash 00_compile_and_generate.sh        # all scales
# or: bash 00_compile_and_generate.sh 0.2   # single scale
```

Uses **dbgen v3.0.0** with fixed seed 12345. Output:

```
code/setup/tpch-data/sf0.2/*.tbl      ~250 MB
code/setup/tpch-data/sf1/*.tbl        ~1.2 GB
code/setup/tpch-data/sf3/*.tbl        ~3.6 GB
```

Then load into PostgreSQL 16 with:

```bash
bash code/setup/02_load_data.sh 0.2
bash code/setup/02_load_data.sh 1
bash code/setup/02_load_data.sh 3
bash code/setup/03_create_base_indexes.sql
```

### SDSS SkyServer

Query log `SkyLog_Workload.csv` is a 19 000-row subset of the SDSS DR16 query log, redistributable under the SDSS data-release license. Redownload from:

<https://skyserver.sdss.org/log/en/traffic/>

Place at `code/experiments/data/SkyLog_Workload.csv`.

### JOB / IMDB (Join Order Benchmark)

Per Leis et al., *"How Good Are Query Optimizers, Really?"* (VLDB 2015). Download the IMDB snapshot (≈2 GB `imdb.tgz`) and the 113 JOB queries from:

<http://homepages.cwi.nl/~boncz/job/imdb.tgz>
<https://github.com/gregrahn/join-order-benchmark>

Extract under `code/experiments/data/imdb/` and the JOB SQL files under `code/experiments/data/job/`.

### pgbench (TPC-B)

Built into PostgreSQL. Initialise with:

```bash
pgbench -i -s 10 pgbench_db
```

### MongoDB 7 (Cross-engine validation)

Four polymorphic collections merged into a single `combined_clean` collection (5 M documents). Scripts in `code/experiments/mongo_ce/` (if present) or derive from the A-CE description in `supplementary/supplementary.tex §XIV`.

## Data licences

| Dataset        | License                                                                 |
|----------------|-------------------------------------------------------------------------|
| TPC-H          | Redistributable per TPC License; we distribute only processed summaries  |
| SDSS SkyServer | CC-BY 4.0 per SDSS data-release policy                                   |
| JOB / IMDB     | Per original authors; IMDB subject to IMDB Non-Commercial Terms          |
| pgbench        | PostgreSQL License (synthetic, no external data)                         |

## Checksums

Processed CSVs are stable across platforms. Reference SHA-256 for the main summary file:

```
summary_stats.csv   67b70fdd4f1a51e4019eaf7982e71aed632f62c7d96d8c59d63476b25732a565
```
