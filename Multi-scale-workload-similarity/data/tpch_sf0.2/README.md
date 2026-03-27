# TPC-H SF=0.2 Data Directory

Place your experiment trace files here as:

```
run_01/trace.csv
run_02/trace.csv
run_03/trace.csv
run_04/trace.csv
run_05/trace.csv
```

Each `trace.csv` must have these columns:
```
run, window, phase, seq, query, op_type, exec_ms, ok, err,
planning_ms, exec_ms_pg, rows_returned, shared_blks_hit, shared_blks_read,
temp_blks_written
```

To generate: follow `STEP1_setup_tpch.md` in the tpch_experiment folder,
then run `STEP2_run_workload.py` on your PostgreSQL instance.

For SF > 0.2: see `experiments/03_scale_tpch.py --sf 1 3 10`
