-- =============================================================
-- Base Index Setup for TPC-H Experiments
-- HSM Throughput Experiments
--
-- DESIGN RATIONALE (v2 — corrected for experimental validity):
-- ─────────────────────────────────────────────────────────────
-- The baseline index set contains ONLY Foreign Key / join indexes.
-- Filter-predicate indexes (l_shipdate, c_mktsegment, p_type, etc.)
-- have been intentionally excluded.
--
-- Why this matters:
--   In v1 the baseline carried 30 indexes (9 FK + 21 filter).  The
--   advisor then had nothing to add — every filter column was already
--   indexed — so all four conditions were effectively "no advisor".
--   The experiment measured only scheduling overhead, not index benefit.
--
--   With FK-only base indexes, queries that filter on l_shipdate,
--   c_mktsegment, p_type, etc. revert to sequential scans, restoring
--   genuine O(N) scaling with SF.  The advisor can now create indexes
--   with real incremental benefit, and HSM-gated vs periodic vs
--   always_on vs baseline produce meaningfully different QPS values.
--
-- FK indexes are kept because they are structurally required for
-- JOIN correctness and planning: without them the planner switches
-- to nested-loop × hash-join plans that are pathologically slow,
-- and these indexes are never "discovered" by a column-frequency
-- advisor (the advisor targets filter columns, not join columns).
--
-- All 22 TPC-H queries are covered across 3 workload phases:
--   Phase A:     Q1, Q3, Q4, Q6, Q12   (lineitem / orders heavy)
--   Phase B:     Q10, Q13, Q14, Q17, Q18  (customer / part heavy)
--   Phase C:     Q2, Q8, Q11             (mixed analytical)
-- =============================================================

-- ── Foreign Key / join indexes (9 total) ────────────────────────────────────
-- lineitem join spine
CREATE INDEX IF NOT EXISTS idx_lineitem_orderkey   ON lineitem  (l_orderkey);
CREATE INDEX IF NOT EXISTS idx_lineitem_partkey    ON lineitem  (l_partkey);
CREATE INDEX IF NOT EXISTS idx_lineitem_suppkey    ON lineitem  (l_suppkey);

-- orders join spine
CREATE INDEX IF NOT EXISTS idx_orders_custkey      ON orders    (o_custkey);

-- partsupp join spine (composite supplier-part lookup)
CREATE INDEX IF NOT EXISTS idx_partsupp_partkey    ON partsupp  (ps_partkey);
CREATE INDEX IF NOT EXISTS idx_partsupp_suppkey    ON partsupp  (ps_suppkey);

-- dimension table FK indexes
CREATE INDEX IF NOT EXISTS idx_customer_nationkey  ON customer  (c_nationkey);
CREATE INDEX IF NOT EXISTS idx_supplier_nationkey  ON supplier  (s_nationkey);
CREATE INDEX IF NOT EXISTS idx_nation_regionkey    ON nation    (n_regionkey);

-- ── Refresh statistics on all tables ────────────────────────────────────────
ANALYZE lineitem, orders, customer, part, partsupp, supplier, nation, region;

\echo 'Base indexes (FK/join only, v2) created and statistics updated.'
