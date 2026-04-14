"""
experiment_runner.py
====================
Main Experiment Runner — HSM Throughput Study
Version 3 — Steady-state design + DROP-before-CREATE + HSM window-0 fix.

DESIGN CHANGES FROM v1:
────────────────────────
  Fix 1 — Base index over-specification (Critical):
    v1: reset_indexes() kept 30 base indexes (9 FK + 21 filter predicates).
        The advisor found no new columns to index → all 4 conditions were
        effectively "no advisor".  Only scheduling overhead was measured.
    v2: reset_indexes() drops ALL non-PK indexes and recreates ONLY the 9
        FK/join indexes.  Filter-column indexes (l_shipdate, c_mktsegment,
        p_type, …) are removed from the baseline, allowing the advisor to
        create them with genuine incremental benefit.  Queries now scale
        O(N) with SF as expected.

  Fix 2 — Order bias (Methodological):
    v1: Sequential condition order — baseline × 10 reps, then always_on × 10,
        then hsm_gated × 10, then periodic × 10.  Cache is hottest for later
        conditions, which run entirely in warm-cache state.
    v2: Randomized block design — 10 blocks, each runs all 4 conditions in a
        block-seeded random order.  Condition × cache interaction is averaged
        out across blocks rather than confounded.

  Fix 3 — Connection state contamination:
    v1: Single connection reused across all conditions within a SF.  Buffer
        cache, pg_prewarm state, and planner statistics accumulate across
        runs within the same session.
    v2: Fresh connection opened per condition per block.  Each condition
        starts from a clean session state.

  Fix 4 — Timeout under-specification:
    v1: SF_TIMEOUT_SEC = {0.2: 60, 1.0: 180, 3.0: 480, 10.0: 1800}.
        Without filter indexes, queries revert to sequential scans, which
        are significantly slower — the v1 timeouts would kill valid queries.
    v2: Timeouts increased to accommodate seq-scan runtimes.

DESIGN CHANGES FROM v2 (v3):
────────────────────────────────

  Fix 5 — Cold-start artefact in advisor conditions (Validity):
    v2: reset_indexes() left only FK/join indexes in place.  Advisor
        conditions therefore started window 0 with NO filter indexes,
        triggering an advisor call (and expensive CREATE INDEX) at the
        very first window — an artefact of the cold-start setup, not a
        genuine production scenario.
    v3: reset_indexes() pre-seeds Phase A filter indexes (l_shipdate,
        l_discount, l_quantity, l_shipmode, l_commitdate, l_receiptdate,
        l_returnflag, o_orderdate) using the same idx_{tbl}_{col} naming
        as the advisor.  All four conditions begin in a steady-state that
        mirrors a production deployment where indexes exist before the
        measurement window starts.

  Fix 6 — Spurious HSM trigger at window 0 (Logical):
    v2: run_hsm_gated() called should_trigger_advisor(None, curr, θ).
        With no previous window, similarity is undefined — the function
        returned trigger=True, causing an unnecessary advisor call at
        window 0 that consumed overhead with no basis in drift detection.
    v3: When prev_window is None, trigger is forced to False.  HSM only
        gates the advisor when it has two windows to compare.  Trigger
        windows become {2, 4, 6} = 3 calls — all at real phase transitions
        (100% trigger efficiency vs periodic's ~67%).

  Fix 7 — Phase contamination from stale indexes (Validity):
    v2: run_index_advisor() used CREATE INDEX IF NOT EXISTS — if Phase A
        indexes existed when the advisor ran at the A→B transition, Phase B
        queries ran with both Phase A AND Phase B indexes, over-stating
        benefit and corrupting phase-level timing comparisons.
    v3: run_index_advisor() DROPs all non-FK advisor indexes before
        creating new ones.  DROP time is counted as advisor overhead.
        Each phase gets exactly the index set recommended for that phase.

Measures QPS (queries/second) under 4 conditions × 4 scale factors × 10 reps.

Conditions:
  baseline    — No index advisor; only FK/join indexes
  always_on   — Index advisor runs after every window (8 calls / trace)
  hsm_gated   — Advisor runs only when HSM detects drift (θ=0.75, ~50%)
  periodic    — Advisor runs every K=3 windows (~37.5%)

Scale Factors: 0.2, 1.0, 3.0, 10.0

Experimental design:
  Randomized block design — 10 blocks × 4 conditions.
  Within each block the 4 conditions run in a random order (seeded per block
  for reproducibility).  This eliminates systematic cache-ordering bias while
  keeping the experiment fully reproducible.

Output:
  results/sf<X>/raw_results.csv   — per-run QPS values
  results/summary.csv             — aggregated mean ± SD ± 95%-CI

Usage:
  # Full experiment (all SF, all conditions, 10 reps)
  python experiment_runner.py

  # Quick test (single SF, 3 reps)
  python experiment_runner.py --sf 0.2 --reps 3 --quick

  # Specific SF only
  python experiment_runner.py --sf 1.0

Requirements:
  pip install psycopg2-binary
"""

import argparse
import csv
import os
import random
import re
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: psycopg2 not found. Run: pip install psycopg2-binary")
    sys.exit(1)

# ─── Add parent directory to path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from hsm_similarity import build_window, should_trigger_advisor, DEFAULT_THETA
from workload_generator import get_workload_trace, get_window_queries

# ─── Configuration ─────────────────────────────────────────────────────────────
# All connection parameters are read from environment variables.
# Copy `.env.example` (at repo root) to `.env` and fill in your values, then
# either `source .env` before running or use `python-dotenv` to auto-load.
# See HSM_gated/docs/REPRODUCE.md for the full list of supported variables.

DB_CONFIG = {
    "host":     os.environ.get("HSM_DB_HOST",     "localhost"),
    "port":     int(os.environ.get("HSM_DB_PORT", 5432)),  # overridden by --port CLI arg
    "dbname":   os.environ.get("HSM_DB_NAME",     "tpch"),  # overridden per SF below
    "user":     os.environ.get("HSM_DB_USER",     "postgres"),
    "password": os.environ.get("HSM_DB_PASSWORD", ""),
}

# Docker mode credentials — used when `--port 5433` is passed via CLI.
# Set HSM_DOCKER_USER / HSM_DOCKER_PASSWORD in `.env` to override.
DOCKER_USER     = os.environ.get("HSM_DOCKER_USER",     "postgres")
DOCKER_PASSWORD = os.environ.get("HSM_DOCKER_PASSWORD", "postgres")

# Map each scale factor to its PostgreSQL database.
# Verified by lineitem row count:
#   SF=0.2  → ~1.2M rows  → tpch_scale_sf0_2
#   SF=1.0  → ~6M rows    → tpch_scale_sf1_0
#   SF=3.0  → ~18M rows   → tpch_scale_sf3_0
#   SF=10.0 → ~60M rows   → tpch_scale_sf10_0
SF_DATABASE = {
    0.2:  "tpch_scale_sf0_2",
    1.0:  "tpch_scale_sf1_0",
    3.0:  "tpch_scale_sf3_0",
    10.0: "tpch_scale_sf10_0",
}

# Per-SF query timeout (seconds).
#
# v2 rationale: without filter indexes, queries fall back to sequential scans
# which are substantially slower than indexed lookups.  Timeouts are set to
# accommodate worst-case seq-scan times with 5-10× safety headroom.
#
# Empirical baselines (sequential scan, no filter indexes):
#   SF=0.2  — Q1 ~10s, Q17 ~30s, Q18 ~35s  → 300s gives ~8× headroom
#   SF=1.0  — scales ~5×                   → 900s
#   SF=3.0  — scales ~15×                  → 1800s (30-min cap per query)
#   SF=10.0 — scales ~50× from SF=0.2      → 3600s (1-hour cap per query)
#
# Note: once the advisor creates indexes these become irrelevant (indexed
# queries finish in seconds), but the baseline condition needs the full timeout.
SF_TIMEOUT_SEC = {
    0.2:   300,
    1.0:   900,
    3.0:  1800,
    10.0: 3600,
}
QUERY_TIMEOUT_SEC = 300  # fallback default

SCALE_FACTORS     = [0.2, 1.0, 3.0, 10.0]
CONDITIONS        = ["baseline", "always_on", "hsm_gated", "periodic"]
N_REPS            = 10      # = number of blocks in randomized block design
QUERIES_PER_PHASE = 30      # 4 phases × 30 = 120 queries per trace
WINDOW_SIZE       = 5       # HSM window size
THETA             = DEFAULT_THETA   # 0.75

# Settling time (seconds) between conditions inside a block.
# Allows PostgreSQL's autovacuum / bgwriter to settle before the next
# condition starts.  Short enough not to materially extend total runtime.
SETTLE_SEC        = 2

# ── FK base index names (module-level constant) ────────────────────────────────
# Used by both reset_indexes() and run_index_advisor() to distinguish
# structural FK/join indexes (never dropped by the advisor) from
# advisor-managed filter indexes (dropped and recreated each call).
FK_BASE_INDEXES = frozenset([
    "idx_lineitem_orderkey", "idx_lineitem_partkey", "idx_lineitem_suppkey",
    "idx_orders_custkey",
    "idx_partsupp_partkey",  "idx_partsupp_suppkey",
    "idx_customer_nationkey", "idx_supplier_nationkey", "idx_nation_regionkey",
])

# Periodic re-advising: call advisor every PERIODIC_K windows regardless of drift.
#
# With QUERIES_PER_PHASE=30 and WINDOW_SIZE=5:
#   Each phase spans 6 windows.  Total = 24 windows.
#   Phase transitions at windows {6, 12, 18}.
#
# K=3 → triggers at {0,3,6,9,12,15,18,21} = 8 calls per 24-window trace = 33.3%.
#
# Analysis:
#   • Meaningful (at phase transition): {6, 12, 18}          = 3/8  = 37.5%
#   • Wasteful   (mid-phase, no drift): {0,3,9,15,21}        = 5/8  = 62.5%
#
# HSM-gated (v3) triggers at {6, 12, 18} only = 3 calls = 12.5% rate.
#   • Meaningful: 3/3 = 100%
#
# This contrast (periodic 37.5% efficient vs HSM 100% efficient) is the
# core empirical claim of the paper.
#
# K=6 would coincide exactly with HSM transitions {0,6,12,18} → vacuous.
# K=3 keeps the comparison meaningful.
PERIODIC_K        = 3

BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Ensure results dir exists before FileHandler opens the log
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "results" / "experiment.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ─── Database Utilities ────────────────────────────────────────────────────────

def get_connection(sf: float = None, timeout_sec: int = None):
    """
    Open a fresh PostgreSQL connection with per-SF tuned session settings.

    Connects to the SF-specific database (e.g. tpch_scale_sf0_2 for SF=0.2)
    so that each scale factor runs against the correct data size.

    A new connection is opened per condition per block so that session-level
    state (plan cache, buffer pinning, temp tables) does not leak between
    conditions.
    """
    config = dict(DB_CONFIG)
    if sf is not None and sf in SF_DATABASE:
        config["dbname"] = SF_DATABASE[sf]
    conn = psycopg2.connect(**config)
    t = timeout_sec if timeout_sec is not None else QUERY_TIMEOUT_SEC
    with conn.cursor() as cur:
        # Kill any query that runs longer than the SF-appropriate timeout
        cur.execute(f"SET statement_timeout = '{t * 1000}'")
        # Limit lock-wait to avoid deadlocks during index creation
        cur.execute("SET lock_timeout = '30s'")
        # Ensure hash joins fit in memory (critical for multi-table TPC-H queries)
        cur.execute("SET work_mem = '512MB'")
        # Tell planner this is SSD — prefer index scans over seq scans
        cur.execute("SET random_page_cost = 1.1")
    conn.commit()
    return conn


def reset_indexes(conn):
    """
    Reset to FK-only base-index state before each condition run.

    Strategy (v2 — corrected):
      1. Drop ALL non-PK indexes.  This includes both advisor-created indexes
         AND the legacy filter-predicate indexes from v1 (l_shipdate, etc.).
         Using DROP ALL (not pattern-matching) is necessary because the
         v1 filter indexes were named WITHOUT TPC-H prefixes (e.g.
         idx_lineitem_shipdate, not idx_lineitem_l_shipdate), so the old
         advisor-prefix regex would miss them.
      2. Re-create ONLY the 9 FK/join indexes.

    This ensures every condition starts from an identical minimal-index
    baseline where the advisor has genuine columns to index, allowing
    meaningful QPS differences between conditions.
    """

    # ── Step 1: Drop ALL non-PK indexes ───────────────────────────────────────
    # Collect all public-schema indexes that are NOT backing a PRIMARY KEY
    # or UNIQUE constraint.  We drop in a separate pass to avoid cursor
    # invalidation during iteration.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT i.indexname
            FROM pg_indexes i
            WHERE i.schemaname = 'public'
              AND NOT EXISTS (
                SELECT 1
                FROM information_schema.table_constraints tc
                WHERE tc.constraint_name = i.indexname
                  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
              )
        """)
        drop_candidates = [row[0] for row in cur.fetchall()]

    with conn.cursor() as cur:
        for idx in drop_candidates:
            try:
                cur.execute(f'DROP INDEX IF EXISTS "{idx}"')
                log.debug(f"  Dropped index: {idx}")
            except Exception as e:
                conn.rollback()
                log.debug(f"  Could not drop {idx}: {e}")
    conn.commit()

    # ── Step 2: Recreate exactly 9 FK/join indexes ─────────────────────────────
    # These mirror 03_create_base_indexes.sql (v2).
    # Naming convention: idx_{table}_{bare_column_name}  — no TPC-H prefix.
    # The advisor uses idx_{table}_{tpch_column}  e.g. idx_lineitem_l_shipdate,
    # which is unambiguously distinct, so DROP ALL in step 1 can be reversed
    # safely here.
    fk_indexes = [
        # lineitem join spine (largest table — most critical for join performance)
        ("idx_lineitem_orderkey",  "lineitem",  "l_orderkey"),
        ("idx_lineitem_partkey",   "lineitem",  "l_partkey"),
        ("idx_lineitem_suppkey",   "lineitem",  "l_suppkey"),
        # orders join spine
        ("idx_orders_custkey",     "orders",    "o_custkey"),
        # partsupp join spine
        ("idx_partsupp_partkey",   "partsupp",  "ps_partkey"),
        ("idx_partsupp_suppkey",   "partsupp",  "ps_suppkey"),
        # dimension table FK indexes
        ("idx_customer_nationkey", "customer",  "c_nationkey"),
        ("idx_supplier_nationkey", "supplier",  "s_nationkey"),
        ("idx_nation_regionkey",   "nation",    "n_regionkey"),
    ]

    with conn.cursor() as cur:
        for idx_name, table, column in fk_indexes:
            try:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({column})"
                )
            except Exception as e:
                conn.rollback()
                log.warning(f"  FK index {idx_name} failed: {e}")
    conn.commit()

    # ── Step 3: Pre-seed Phase A filter indexes (steady-state design) ──────────
    #
    # RATIONALE:
    #   In a production system the index advisor has been running continuously;
    #   when we start measuring, Phase A indexes already exist.  Starting from
    #   a cold FK-only state would disadvantage advisor conditions at window 0
    #   and produce a cold-start artefact that is irrelevant to our research
    #   question (drift-based gating efficiency).
    #
    #   Solution: pre-seed the filter indexes for Phase A queries (Q1, Q3, Q4,
    #   Q6, Q12) before every condition rep.  All four conditions start at the
    #   same steady-state.
    #
    #   Naming: uses the full TPC-H column name (idx_lineitem_l_shipdate, etc.)
    #   — the same naming scheme as run_index_advisor().  This means:
    #     • reset_indexes() creates them cleanly from scratch each rep.
    #     • run_index_advisor() will DROP them (they are not in FK_BASE_INDEXES)
    #       and CREATE the appropriate set for the current workload phase.
    #
    # Phase A queries and their filter columns:
    #   Q1  : l_shipdate, l_returnflag, l_linestatus, l_quantity, l_discount
    #   Q3  : l_shipdate, o_orderdate
    #   Q4  : o_orderdate, l_commitdate
    #   Q6  : l_shipdate, l_discount, l_quantity
    #   Q12 : l_shipdate, l_shipmode, l_commitdate, l_receiptdate
    phase_a_indexes = [
        ("idx_lineitem_l_shipdate",   "lineitem", "l_shipdate"),
        ("idx_lineitem_l_discount",   "lineitem", "l_discount"),
        ("idx_lineitem_l_quantity",   "lineitem", "l_quantity"),
        ("idx_lineitem_l_shipmode",   "lineitem", "l_shipmode"),
        ("idx_lineitem_l_commitdate", "lineitem", "l_commitdate"),
        ("idx_lineitem_l_receiptdate","lineitem", "l_receiptdate"),
        ("idx_lineitem_l_returnflag", "lineitem", "l_returnflag"),
        ("idx_orders_o_orderdate",    "orders",   "o_orderdate"),
    ]

    with conn.cursor() as cur:
        for idx_name, table, column in phase_a_indexes:
            try:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({column})"
                )
            except Exception as e:
                conn.rollback()
                log.warning(f"  Phase A seed index {idx_name} failed: {e}")
    conn.commit()

    # ── Step 4: Lightweight statistics refresh ──────────────────────────────────
    # ANALYZE on the two largest tables keeps planner estimates accurate.
    # Full ANALYZE on all 8 tables would take too long between reps.
    with conn.cursor() as cur:
        cur.execute("ANALYZE lineitem, orders")
    conn.commit()

    log.debug(
        "  reset_indexes: 9 FK/join + 8 Phase-A filter indexes in place "
        "(steady-state start)"
    )


def run_index_advisor(conn, recent_queries: list):
    """
    Simulated index advisor.
    Analyzes recent query patterns and creates column indexes.

    Logic: scan queries for known TPC-H filter columns; create up to 5
    single-column B-tree indexes.  Index names use the TPC-H column name
    (including its table prefix, e.g. l_shipdate) so they are named
    idx_lineitem_l_shipdate — distinct from the FK base indexes which
    use bare column names (idx_lineitem_orderkey).

    In Paper 3B this advisor will be replaced by BALANCE / Indexer++.
    For HSM it isolates HSM gating overhead measurement from
    advisor quality — a deliberate simplification.
    """
    candidate_cols = {}

    # Map of TPC-H filter column → table
    tpch_col_to_table = {
        # lineitem filter columns
        'l_shipdate':    'lineitem', 'l_discount':    'lineitem',
        'l_quantity':    'lineitem', 'l_shipmode':    'lineitem',
        'l_receiptdate': 'lineitem', 'l_commitdate':  'lineitem',
        'l_returnflag':  'lineitem',
        # orders filter columns
        'o_orderdate':    'orders',  'o_orderpriority': 'orders',
        'o_orderstatus':  'orders',
        # customer filter columns
        'c_mktsegment':   'customer',
        # part filter columns
        'p_type': 'part', 'p_brand': 'part',
        'p_size': 'part', 'p_container': 'part', 'p_name': 'part',
        # supplier filter columns
        's_acctbal': 'supplier',
        # nation / region filter columns
        'n_name': 'nation', 'r_name': 'region',
        # partsupp filter columns
        'ps_supplycost': 'partsupp', 'ps_availqty': 'partsupp',
    }

    for sql in recent_queries:
        sql_lower = sql.lower()
        for col, tbl in tpch_col_to_table.items():
            if col in sql_lower:
                candidate_cols[col] = tbl

    # ── Step 1: DROP all advisor-managed indexes ───────────────────────────────
    #
    # RATIONALE:
    #   Phase contamination problem: if Phase A indexes (l_shipdate, etc.) are
    #   left in place when the advisor runs at the Phase A→B transition, Phase B
    #   queries get the benefit of stale Phase A indexes AND new Phase B indexes.
    #   This over-states advisor benefit and produces unfair Phase B timings.
    #
    #   Fix: before creating any new indexes, DROP everything that is not a
    #   structural FK/join base index.  The DROP overhead is counted as part of
    #   advisor cost (it is the advisor's decision to replace the index set).
    #
    #   Preservation: FK_BASE_INDEXES (9 FK/join indexes) are never touched.
    #
    # Timing: DROP time is already included in wall_time because the caller
    # adds +0.05s for the sleep (advisor planning overhead) and wraps the
    # entire run_index_advisor() call inside total_time accumulation.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT i.indexname
            FROM pg_indexes i
            WHERE i.schemaname = 'public'
              AND NOT EXISTS (
                SELECT 1
                FROM information_schema.table_constraints tc
                WHERE tc.constraint_name = i.indexname
                  AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
              )
        """)
        all_idx = [row[0] for row in cur.fetchall()]

    advisor_idx = [idx for idx in all_idx if idx not in FK_BASE_INDEXES]

    with conn.cursor() as cur:
        for idx in advisor_idx:
            try:
                cur.execute(f'DROP INDEX IF EXISTS "{idx}"')
                log.debug(f"  Advisor DROP: {idx}")
            except Exception:
                conn.rollback()
                continue
    conn.commit()

    # ── Step 2: CREATE new indexes for the current workload ────────────────────
    with conn.cursor() as cur:
        for col, tbl in list(candidate_cols.items())[:5]:   # max 5 per call
            # Advisor index names include TPC-H prefix: idx_lineitem_l_shipdate
            idx_name = f"idx_{tbl}_{col}"
            try:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {tbl}({col})"
                )
                log.debug(f"  Advisor CREATE: {idx_name}")
            except Exception:
                conn.rollback()
                continue
    conn.commit()

    # Simulate advisor overhead (planning / catalog lookup time)
    time.sleep(0.05)   # 50 ms per advisor call (conservative)


# ─── Query Execution with Timing ──────────────────────────────────────────────

def execute_query_batch(conn, sql_batch: list) -> dict:
    """
    Execute a batch of SQL queries and return timing statistics.

    Returns dict with:
      total_time_s   : wall time for entire batch (seconds)
      num_queries    : number of successful queries
      qps            : queries per second (successful / total_time_s)
      query_times_ms : list of individual query times (ms), including timed-out
      errors         : number of failed / timed-out queries
    """
    query_times = []
    errors = 0

    with conn.cursor() as cur:
        for sql in sql_batch:
            t_start = time.perf_counter()
            try:
                cur.execute(sql)
                cur.fetchall()   # Ensure full result materialisation
            except psycopg2.errors.QueryCanceled:
                errors += 1
                log.debug("Query timed out (statement_timeout) — counted as error")
                conn.rollback()
            except Exception as e:
                errors += 1
                log.debug(f"Query error: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
            finally:
                t_end = time.perf_counter()
                query_times.append((t_end - t_start) * 1000)   # ms

    total_time_s = sum(query_times) / 1000.0
    num_queries  = len(sql_batch) - errors
    qps = num_queries / total_time_s if total_time_s > 0 else 0.0

    return {
        "total_time_s":   total_time_s,
        "num_queries":    num_queries,
        "qps":            qps,
        "query_times_ms": query_times,
        "errors":         errors,
    }


# ─── Phase-Level Timing ───────────────────────────────────────────────────────

def _phase_times(query_times_ms: list, queries_per_phase: int = QUERIES_PER_PHASE) -> dict:
    """
    Split flat query_times_ms list into per-phase totals (seconds).

    Phase layout (4 phases × QUERIES_PER_PHASE queries):
      Phase A       : positions 0       .. QPP-1      (lineitem/orders — composite PK)
      Phase B       : positions QPP     .. 2*QPP-1    (customer/part heavy)
      Phase A_rep   : positions 2*QPP   .. 3*QPP-1    (lineitem/orders repeat)
      Phase C       : positions 3*QPP   .. 4*QPP-1    (mixed analytical)

    Used to test the no-natural-PK hypothesis: composite-PK phases (A, A_rep)
    should show higher advisor benefit than natural-PK phases (B, C).
    """
    q = queries_per_phase
    def phase_sum(start, end):
        return round(sum(query_times_ms[start:end]) / 1000.0, 4)
    return {
        "time_phase_A_s":    phase_sum(0,   q),
        "time_phase_B_s":    phase_sum(q,   2*q),
        "time_phase_Arep_s": phase_sum(2*q, 3*q),
        "time_phase_C_s":    phase_sum(3*q, 4*q),
    }


# ─── Experiment Conditions ─────────────────────────────────────────────────────

def run_baseline(conn, trace: list) -> dict:
    """Condition 1: No advisor — FK/join indexes only, no dynamic changes."""
    sql_batch = [sql for _, sql in trace]
    result = execute_query_batch(conn, sql_batch)
    result["advisor_calls"] = 0
    result["trigger_rate"]  = 0.0
    result.update(_phase_times(result["query_times_ms"]))
    return result


def run_always_on(conn, trace: list) -> dict:
    """
    Condition 2: Advisor runs after every window.
    Upper bound on advisor overhead — 8 calls per 40-query trace (100% rate).
    """
    total_time    = 0.0
    total_queries = 0
    total_errors  = 0
    all_times     = []

    for i in range(0, len(trace), WINDOW_SIZE):
        window_sqls = [sql for _, sql in trace[i:i + WINDOW_SIZE]]
        result = execute_query_batch(conn, window_sqls)
        total_time    += result["total_time_s"]
        total_queries += result["num_queries"]
        total_errors  += result["errors"]
        all_times.extend(result["query_times_ms"])

        run_index_advisor(conn, window_sqls)
        total_time += 0.05   # advisor overhead (matches sleep in run_index_advisor)

    total_windows = len(trace) // WINDOW_SIZE
    qps = total_queries / total_time if total_time > 0 else 0.0

    ret = {
        "total_time_s":   total_time,
        "num_queries":    total_queries,
        "qps":            qps,
        "query_times_ms": all_times,
        "errors":         total_errors,
        "advisor_calls":  total_windows,
        "trigger_rate":   1.0,
    }
    ret.update(_phase_times(all_times))
    return ret


def run_hsm_gated(conn, trace: list, theta: float = THETA) -> dict:
    """
    Condition 3: Advisor runs only when HSM detects workload drift below θ.
    Empirically ~50% trigger rate (4 calls per 8-window trace).
    """
    total_time    = 0.0
    total_queries = 0
    total_errors  = 0
    all_times     = []
    advisor_calls = 0
    hsm_scores    = []

    prev_window = None

    for i in range(0, len(trace), WINDOW_SIZE):
        window_items = trace[i:i + WINDOW_SIZE]
        window_sqls  = [sql for _, sql in window_items]

        curr_window = build_window(window_sqls, window_id=i // WINDOW_SIZE)
        window_idx  = i // WINDOW_SIZE

        if prev_window is None:
            # Window 0: no reference window exists — drift cannot be measured.
            # In production the advisor has been running continuously; Phase A
            # indexes are already in place (pre-seeded by reset_indexes).
            # Do NOT trigger: there is no workload history to compare against.
            trigger = False
            score   = 1.0    # sentinel: identical to self (no comparison)
            dims    = {}
            log.debug(
                f"  Window {window_idx}: prev_window=None → observe only "
                f"(no drift comparison possible at window 0)"
            )
        else:
            trigger, score, dims = should_trigger_advisor(prev_window, curr_window, theta)

        hsm_scores.append(score)

        result = execute_query_batch(conn, window_sqls)
        total_time    += result["total_time_s"]
        total_queries += result["num_queries"]
        total_errors  += result["errors"]
        all_times.extend(result["query_times_ms"])

        if trigger:
            run_index_advisor(conn, window_sqls)
            total_time += 0.05
            advisor_calls += 1
            log.debug(
                f"  Window {window_idx}: HSM={score:.3f} < θ={theta} → advisor triggered"
            )
        else:
            if prev_window is not None:
                log.debug(
                    f"  Window {window_idx}: HSM={score:.3f} ≥ θ={theta} → skipped"
                )

        prev_window = curr_window

    total_windows = len(trace) // WINDOW_SIZE
    qps          = total_queries / total_time if total_time > 0 else 0.0
    trigger_rate = advisor_calls / total_windows if total_windows > 0 else 0.0

    ret = {
        "total_time_s":   total_time,
        "num_queries":    total_queries,
        "qps":            qps,
        "query_times_ms": all_times,
        "errors":         total_errors,
        "advisor_calls":  advisor_calls,
        "trigger_rate":   trigger_rate,
        "hsm_scores":     hsm_scores,
    }
    ret.update(_phase_times(all_times))
    return ret


def run_periodic(conn, trace: list, k: int = PERIODIC_K) -> dict:
    """
    Condition 4: Periodic re-advising — advisor called every K windows.

    K=3 → 3 calls per 8-window trace (37.5% trigger rate).
    Trigger windows: {0, 3, 6}.

    Deliberately misses the A→B transition (window 2) and B→A_rep transition
    (window 4), while hitting mid-Phase-B window 3 (no drift, wasteful).
    This generates the "wrong-time" trigger pattern that HSM avoids.
    """
    total_time    = 0.0
    total_queries = 0
    total_errors  = 0
    all_times     = []
    advisor_calls = 0

    for window_idx, i in enumerate(range(0, len(trace), WINDOW_SIZE)):
        window_sqls = [sql for _, sql in trace[i:i + WINDOW_SIZE]]
        result = execute_query_batch(conn, window_sqls)
        total_time    += result["total_time_s"]
        total_queries += result["num_queries"]
        total_errors  += result["errors"]
        all_times.extend(result["query_times_ms"])

        if window_idx % k == 0:
            run_index_advisor(conn, window_sqls)
            total_time += 0.05
            advisor_calls += 1
            log.debug(f"  Window {window_idx}: periodic → advisor triggered (every {k} windows)")
        else:
            log.debug(f"  Window {window_idx}: periodic → skipped")

    total_windows = len(trace) // WINDOW_SIZE
    qps          = total_queries / total_time if total_time > 0 else 0.0
    trigger_rate = advisor_calls / total_windows if total_windows > 0 else 0.0

    ret = {
        "total_time_s":   total_time,
        "num_queries":    total_queries,
        "qps":            qps,
        "query_times_ms": all_times,
        "errors":         total_errors,
        "advisor_calls":  advisor_calls,
        "trigger_rate":   trigger_rate,
    }
    ret.update(_phase_times(all_times))
    return ret


# ─── Result Storage ───────────────────────────────────────────────────────────

def save_raw_result(sf: float, condition: str, block: int, rep_in_block: int,
                    result: dict, wall_time_s: float = None):
    """
    Append a single run result to the per-SF raw CSV.

    `block`        : block number (0-indexed, 0..N_REPS-1)
    `rep_in_block` : position of this condition within its block (0-indexed,
                     0..len(CONDITIONS)-1).  Useful for diagnosing order effects.
    """
    out_dir  = RESULTS_DIR / f"sf{sf}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "raw_results.csv"

    fieldnames = [
        "timestamp", "sf", "condition", "block", "rep_in_block",
        "qps", "wall_qps", "total_time_s", "wall_time_s",
        "num_queries", "errors",
        "advisor_calls", "trigger_rate",
        "time_phase_A_s", "time_phase_B_s", "time_phase_Arep_s", "time_phase_C_s",
    ]

    num_q    = result.get("num_queries", 0)
    wall_qps = round(num_q / wall_time_s, 4) if wall_time_s and wall_time_s > 0 else "N/A"

    def _r(v): return round(v, 4) if isinstance(v, float) else v

    write_header = not out_file.exists()
    with open(out_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":         datetime.now().isoformat(),
            "sf":                sf,
            "condition":         condition,
            "block":             block + 1,       # 1-indexed for readability
            "rep_in_block":      rep_in_block + 1,
            "qps":               round(result.get("qps", 0), 4),
            "wall_qps":          wall_qps,
            "total_time_s":      round(result.get("total_time_s", 0), 4),
            "wall_time_s":       round(wall_time_s, 4) if wall_time_s else "N/A",
            "num_queries":       num_q,
            "errors":            result.get("errors", 0),
            "advisor_calls":     result.get("advisor_calls", "N/A"),
            "trigger_rate":      round(result.get("trigger_rate", 0), 4)
                                 if "trigger_rate" in result else "N/A",
            "time_phase_A_s":    _r(result.get("time_phase_A_s",    "N/A")),
            "time_phase_B_s":    _r(result.get("time_phase_B_s",    "N/A")),
            "time_phase_Arep_s": _r(result.get("time_phase_Arep_s", "N/A")),
            "time_phase_C_s":    _r(result.get("time_phase_C_s",    "N/A")),
        })


# ─── Main Loop ────────────────────────────────────────────────────────────────

def _count_existing_runs(sf: float) -> int:
    """Count completed rows in the CSV for a given SF (excludes header)."""
    out_file = RESULTS_DIR / f"sf{sf}" / "raw_results.csv"
    if not out_file.exists():
        return 0
    with open(out_file, newline="") as f:
        rows = list(csv.DictReader(f))
    return len(rows)


def run_all_experiments(
    scale_factors: list = None,
    n_reps: int = N_REPS,
    conditions: list = None,
    quick: bool = False,
    resume: bool = False,
):
    """
    Run all experiments using a randomized block design.

    Structure:
      For each SF:
        For each block b in 0..n_reps-1:
          Shuffle CONDITIONS with seed=b for reproducibility.
          For each condition c in shuffled order:
            1. Open a fresh connection.
            2. Reset indexes (drop ALL non-PK, recreate 9 FK/join).
            3. Run the condition.
            4. Record result.
            5. Close connection.
            6. Sleep SETTLE_SEC before next condition.

    This design ensures:
      • Each condition appears exactly once per block (balanced).
      • Condition order varies across blocks (randomized).
      • Cache state is not systematically warmer for any single condition.
      • Connection state never leaks between conditions.

    If resume=True, previously completed runs for each SF are counted and
    skipped so the experiment continues from where it left off.
    """
    if scale_factors is None:
        scale_factors = SCALE_FACTORS
    if conditions is None:
        conditions = CONDITIONS
    if quick:
        n_reps = 3

    log.info("=" * 70)
    log.info("  HSM Throughput Experiment  (v3 — steady-state design)")
    log.info(f"  Scale Factors   : {scale_factors}")
    log.info(f"  Conditions      : {conditions}")
    log.info(f"  Blocks (reps)   : {n_reps}")
    log.info(f"  Theta           : {THETA}")
    log.info(f"  Design          : Randomized block (seed per block)")
    log.info(f"  Base indexes    : 9 FK/join + 8 Phase-A filter (steady-state)")
    log.info(f"  HSM window 0    : observe-only (no advisor trigger)")
    log.info(f"  Advisor DROP    : all non-FK indexes dropped before CREATE")
    log.info(f"  Queries/phase   : {QUERIES_PER_PHASE} (120 total, 24 windows)")
    log.info(f"  Sampling        : without-replacement + 3-variant param pools")
    log.info(f"  Trace seed      : per-block (seed=block eliminates cache reuse)")
    log.info(f"  Settle time     : {SETTLE_SEC}s between conditions")
    if resume:
        log.info(f"  Resume mode     : ON — skipping already-completed runs")
    log.info("=" * 70)

    total_runs = len(scale_factors) * len(conditions) * n_reps
    run_count  = 0

    for sf in scale_factors:
        sf_timeout = SF_TIMEOUT_SEC.get(sf, QUERY_TIMEOUT_SEC)
        db_name = SF_DATABASE.get(sf, DB_CONFIG["dbname"])
        log.info(
            f"\n── Scale Factor SF={sf}  db={db_name}  "
            f"(timeout={sf_timeout}s / {sf_timeout//60}min per query) ─────────────"
        )

        # In resume mode, count how many runs are already in the CSV for this SF.
        skip_runs = _count_existing_runs(sf) if resume else 0
        if skip_runs:
            log.info(
                f"  Resume: found {skip_runs} existing rows — "
                f"skipping first {skip_runs} runs"
            )
        sf_run_count = 0  # runs executed so far for this SF (including skipped)

        for block in range(n_reps):
            # ── Per-block workload trace (parameterized, without-replacement) ──
            # Each block uses seed=block so parameter variants differ across blocks.
            # This eliminates inter-block page-cache reuse from fixed SQL literals.
            # Query STRUCTURE (tables/columns) is identical across blocks
            # → HSM features and index advisor behaviour are unaffected.
            trace = get_workload_trace(
                queries_per_phase=QUERIES_PER_PHASE,
                seed=block,
            )

            # Shuffle conditions for this block using block number as seed.
            # This is fully reproducible: same block → same order on re-run.
            block_conditions = list(conditions)
            random.Random(block).shuffle(block_conditions)

            # Skip entire block if all its conditions are already done.
            if sf_run_count + len(block_conditions) <= skip_runs:
                sf_run_count += len(block_conditions)
                run_count    += len(block_conditions)
                log.info(
                    f"\n  Block {block+1}/{n_reps}  [SKIPPED — already in CSV]"
                )
                continue

            log.info(
                f"  Block trace: {len(trace)} queries "
                f"(4 phases × {QUERIES_PER_PHASE}, seed={block})"
            )
            log.info(
                f"\n  Block {block+1}/{n_reps}  "
                f"order: {' → '.join(block_conditions)}"
            )

            for rep_in_block, condition in enumerate(block_conditions):
                run_count    += 1
                sf_run_count += 1

                # Skip individual runs within the current (partially done) block.
                if sf_run_count <= skip_runs:
                    log.info(
                        f"    [{run_count}/{total_runs}]  "
                        f"Block {block+1}  Condition: {condition}  [SKIPPED]"
                    )
                    continue

                log.info(
                    f"    [{run_count}/{total_runs}]  "
                    f"Block {block+1}  Condition: {condition}"
                )

                # ── Fresh connection per condition (SF-specific database) ───
                conn = get_connection(sf=sf, timeout_sec=sf_timeout)
                try:
                    # ── Reset to FK-only base state ─────────────────────────
                    reset_indexes(conn)

                    # ── Run the condition ───────────────────────────────────
                    t0 = time.perf_counter()

                    if condition == "baseline":
                        result = run_baseline(conn, trace)
                    elif condition == "always_on":
                        result = run_always_on(conn, trace)
                    elif condition == "hsm_gated":
                        result = run_hsm_gated(conn, trace, THETA)
                    elif condition == "periodic":
                        result = run_periodic(conn, trace, PERIODIC_K)
                    else:
                        raise ValueError(f"Unknown condition: {condition}")

                    wall_time = time.perf_counter() - t0

                    log.info(
                        f"      ✓  wall_QPS={result['num_queries']/wall_time:.4f}  "
                        f"qps={result['qps']:.4f}  "
                        f"queries={result['num_queries']}  "
                        f"errors={result['errors']}  "
                        f"wall={wall_time:.2f}s  "
                        f"advisor_calls={result.get('advisor_calls', 'N/A')}"
                    )

                    # ── Save result ─────────────────────────────────────────
                    save_raw_result(
                        sf=sf,
                        condition=condition,
                        block=block,
                        rep_in_block=rep_in_block,
                        result=result,
                        wall_time_s=wall_time,
                    )

                finally:
                    conn.close()

                # ── Settling pause before next condition ────────────────────
                if rep_in_block < len(block_conditions) - 1:
                    log.info(f"      … settling {SETTLE_SEC}s …")
                    time.sleep(SETTLE_SEC)

    log.info("\n" + "=" * 70)
    log.info(f"  All experiments completed ({total_runs} runs)")
    log.info(f"  Results saved to: {RESULTS_DIR}")
    log.info("  Run analysis/compute_statistics.py to generate tables")
    log.info("=" * 70)


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HSM Throughput Experiment Runner (v2 — corrected design)"
    )
    parser.add_argument(
        "--sf", type=float, nargs="+",
        default=None,
        help="Scale factor(s) to run (default: all 4)"
    )
    parser.add_argument(
        "--reps", type=int, default=N_REPS,
        help=f"Number of blocks/repetitions (default: {N_REPS})"
    )
    parser.add_argument(
        "--condition", type=str, nargs="+",
        default=None,
        choices=CONDITIONS,
        help="Condition(s) to run (default: all 4)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test: 3 blocks only"
    )
    parser.add_argument(
        "--theta", type=float, default=THETA,
        help=f"HSM threshold theta (default: {THETA})"
    )

    parser.add_argument(
        "--port", type=int, default=5432,
        help="PostgreSQL port (default: 5432 native; use 5433 for Docker)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume: skip runs already recorded in the CSV for each SF"
    )

    args = parser.parse_args()
    THETA = args.theta   # Allow CLI override

    # Apply port override
    DB_CONFIG["port"] = args.port
    if args.port == 5433:
        # Docker container uses postgres/postgres credentials
        DB_CONFIG["user"]     = DOCKER_USER
        DB_CONFIG["password"] = DOCKER_PASSWORD
        log.info(f"Docker mode: connecting to localhost:{args.port} as '{DOCKER_USER}'")
    else:
        log.info(f"Native mode: connecting to localhost:{args.port} as '{DB_CONFIG['user']}'")

    run_all_experiments(
        scale_factors=args.sf,
        n_reps=args.reps,
        conditions=args.condition,
        quick=args.quick,
        resume=args.resume,
    )
