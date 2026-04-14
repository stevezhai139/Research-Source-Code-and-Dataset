#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# load_data.sh
# Load TPC-H data from native PostgreSQL into Docker container.
# Compatible with bash 3.2+ (macOS default shell).
#
# USAGE:
#   bash load_data.sh              # load all 4 SF
#   bash load_data.sh 0.2 1.0      # load specific SF only
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration (env-var overridable; see ../.env.example) ──────────────────
NATIVE_HOST="${HSM_DB_HOST:-localhost}"
NATIVE_PORT="${HSM_DB_PORT:-5432}"
NATIVE_USER="${HSM_DB_USER:-$(whoami)}"

DOCKER_HOST="${HSM_DOCKER_HOST:-localhost}"
DOCKER_PORT="${HSM_DOCKER_PORT:-5433}"
DOCKER_USER="${HSM_DOCKER_USER:-postgres}"
DOCKER_PASS="${HSM_DOCKER_PASSWORD:-postgres}"

DUMP_DIR="/tmp/tpch_dumps"
mkdir -p "$DUMP_DIR"

# ── SF → database name mapping (no associative array — bash 3.2 compatible) ───
sf_to_db() {
  case "$1" in
    "0.2")  echo "tpch_scale_sf0_2"  ;;
    "1.0")  echo "tpch_scale_sf1_0"  ;;
    "3.0")  echo "tpch_scale_sf3_0"  ;;
    "10.0") echo "tpch_scale_sf10_0" ;;
    *)      echo ""; return 1        ;;
  esac
}

# ── Which SFs to load ─────────────────────────────────────────────────────────
if [ $# -gt 0 ]; then
  SF_LIST=("$@")
else
  SF_LIST=("0.2" "1.0" "3.0" "10.0")
fi

# ── Wait for Docker PostgreSQL to be ready ────────────────────────────────────
echo "Waiting for Docker PostgreSQL on port $DOCKER_PORT..."
for i in $(seq 1 30); do
  if PGPASSWORD="$DOCKER_PASS" pg_isready \
       -h "$DOCKER_HOST" -p "$DOCKER_PORT" -U "$DOCKER_USER" -q 2>/dev/null; then
    echo "Docker PostgreSQL is ready."
    break
  fi
  sleep 2
  if [ "$i" -eq 30 ]; then
    echo "ERROR: Docker PostgreSQL did not start within 60 seconds."
    exit 1
  fi
done

# ── Load each SF ──────────────────────────────────────────────────────────────
for sf in "${SF_LIST[@]}"; do
  NATIVE_DB="$(sf_to_db "$sf")"
  if [ -z "$NATIVE_DB" ]; then
    echo "ERROR: Unknown SF=$sf. Valid values: 0.2 1.0 3.0 10.0"
    exit 1
  fi
  DOCKER_DB="$NATIVE_DB"
  DUMP_FILE="$DUMP_DIR/${NATIVE_DB}.dump"

  echo ""
  echo "══════════════════════════════════════════"
  echo "  SF=$sf  |  $NATIVE_DB"
  echo "══════════════════════════════════════════"

  # ── Step 1: Dump from native PostgreSQL ─────────────────────────────────
  echo "[1/3] Dumping from native PostgreSQL (port $NATIVE_PORT)..."
  START=$(date +%s)
  pg_dump \
    -h "$NATIVE_HOST" \
    -p "$NATIVE_PORT" \
    -U "$NATIVE_USER" \
    -F c \
    -f "$DUMP_FILE" \
    "$NATIVE_DB"
  END=$(date +%s)
  SIZE=$(du -sh "$DUMP_FILE" | cut -f1)
  echo "    Done: $SIZE in $((END-START))s → $DUMP_FILE"

  # ── Step 2: Create database in Docker ───────────────────────────────────
  echo "[2/3] Creating database $DOCKER_DB in Docker..."
  PGPASSWORD="$DOCKER_PASS" psql \
    -h "$DOCKER_HOST" -p "$DOCKER_PORT" -U "$DOCKER_USER" \
    -c "DROP DATABASE IF EXISTS $DOCKER_DB;" postgres
  PGPASSWORD="$DOCKER_PASS" psql \
    -h "$DOCKER_HOST" -p "$DOCKER_PORT" -U "$DOCKER_USER" \
    -c "CREATE DATABASE $DOCKER_DB;" postgres
  echo "    Done."

  # ── Step 3: Restore into Docker ─────────────────────────────────────────
  echo "[3/3] Restoring into Docker PostgreSQL (port $DOCKER_PORT)..."
  START=$(date +%s)
  PGPASSWORD="$DOCKER_PASS" pg_restore \
    -h "$DOCKER_HOST" -p "$DOCKER_PORT" -U "$DOCKER_USER" \
    -d "$DOCKER_DB" \
    --no-owner --no-privileges \
    -j 2 \
    "$DUMP_FILE"
  END=$(date +%s)
  echo "    Done in $((END-START))s"

  # ── Verify row count ─────────────────────────────────────────────────────
  COUNT=$(PGPASSWORD="$DOCKER_PASS" psql \
    -h "$DOCKER_HOST" -p "$DOCKER_PORT" -U "$DOCKER_USER" \
    -d "$DOCKER_DB" -qAt \
    -c "SELECT COUNT(*) FROM lineitem;")
  echo "    Verified: lineitem = $COUNT rows"
  echo "  SF=$sf loaded successfully ✓"
done

# ── Setup FK/join indexes in all loaded databases ─────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  Setting up FK/join indexes (v2)"
echo "══════════════════════════════════════════"

for sf in "${SF_LIST[@]}"; do
  DB="$(sf_to_db "$sf")"
  echo "  $DB..."
  PGPASSWORD="$DOCKER_PASS" psql \
    -h "$DOCKER_HOST" -p "$DOCKER_PORT" -U "$DOCKER_USER" \
    -d "$DB" -c "
      DO \$\$ DECLARE r RECORD;
      BEGIN
        FOR r IN
          SELECT indexname FROM pg_indexes
          WHERE schemaname = 'public'
            AND indexname NOT IN (
              SELECT constraint_name FROM information_schema.table_constraints
              WHERE constraint_type IN ('PRIMARY KEY','UNIQUE'))
        LOOP
          EXECUTE 'DROP INDEX IF EXISTS ' || quote_ident(r.indexname);
        END LOOP;
      END \$\$;
      CREATE INDEX IF NOT EXISTS idx_lineitem_orderkey   ON lineitem  (l_orderkey);
      CREATE INDEX IF NOT EXISTS idx_lineitem_partkey    ON lineitem  (l_partkey);
      CREATE INDEX IF NOT EXISTS idx_lineitem_suppkey    ON lineitem  (l_suppkey);
      CREATE INDEX IF NOT EXISTS idx_orders_custkey      ON orders    (o_custkey);
      CREATE INDEX IF NOT EXISTS idx_partsupp_partkey    ON partsupp  (ps_partkey);
      CREATE INDEX IF NOT EXISTS idx_partsupp_suppkey    ON partsupp  (ps_suppkey);
      CREATE INDEX IF NOT EXISTS idx_customer_nationkey  ON customer  (c_nationkey);
      CREATE INDEX IF NOT EXISTS idx_supplier_nationkey  ON supplier  (s_nationkey);
      CREATE INDEX IF NOT EXISTS idx_nation_regionkey    ON nation    (n_regionkey);
      ANALYZE lineitem, orders, customer, part, partsupp, supplier, nation, region;
    " > /dev/null
  echo "    FK indexes ready ✓"
done

echo ""
echo "══════════════════════════════════════════"
echo "  All done! Docker PostgreSQL is ready."
echo "  Run: python experiment_runner.py --port 5433"
echo "══════════════════════════════════════════"

# Cleanup dumps
echo ""
read -p "Delete dump files from $DUMP_DIR to free disk space? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
  rm -f "$DUMP_DIR"/tpch_scale_*.dump
  echo "Dump files deleted."
fi
