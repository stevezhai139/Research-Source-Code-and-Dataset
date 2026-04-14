#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# load_imdb.sh
# Load IMDB CSV data into Docker PostgreSQL for JOB benchmark (A8b validation).
#
# Prerequisites:
#   - Docker container running: docker compose up -d
#   - IMDB CSV files extracted in: data/ directory
#     (aka_name.csv, cast_info.csv, title.csv, etc.)
#
# Usage:
#   bash load_imdb.sh
#
# After loading, run HSM validation:
#   cd experiments/
#   python hsm_job_validation.py --execute
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

DOCKER_HOST="${HSM_DOCKER_HOST:-localhost}"
DOCKER_PORT="${HSM_DOCKER_PORT:-5433}"
DOCKER_USER="${HSM_DOCKER_USER:-postgres}"
DOCKER_PASS="${HSM_DOCKER_PASSWORD:-postgres}"
DOCKER_DB="${HSM_IMDB_DB:-imdb}"

export PGPASSWORD="$DOCKER_PASS"

echo "═══════════════════════════════════════════════════════════"
echo "  IMDB Data Loader for Docker PostgreSQL"
echo "  Target: ${DOCKER_HOST}:${DOCKER_PORT}/${DOCKER_DB}"
echo "═══════════════════════════════════════════════════════════"

# ── Wait for Docker PostgreSQL ─────────────────────────────────────────────────
echo ""
echo "[1] Waiting for Docker PostgreSQL on port $DOCKER_PORT ..."
for i in $(seq 1 30); do
    if PGPASSWORD="$DOCKER_PASS" pg_isready \
         -h "$DOCKER_HOST" -p "$DOCKER_PORT" -U "$DOCKER_USER" -q 2>/dev/null; then
        echo "    Docker PostgreSQL is ready."
        break
    fi
    sleep 2
    if [ "$i" -eq 30 ]; then
        echo "    ERROR: Docker PostgreSQL not ready after 60s."
        echo "    Start it with: docker compose up -d"
        exit 1
    fi
done

# ── Create IMDB database ───────────────────────────────────────────────────────
echo ""
echo "[2] Creating database '$DOCKER_DB' ..."
PGPASSWORD="$DOCKER_PASS" psql \
    -h "$DOCKER_HOST" -p "$DOCKER_PORT" -U "$DOCKER_USER" \
    -c "DROP DATABASE IF EXISTS $DOCKER_DB;" postgres 2>/dev/null || true
PGPASSWORD="$DOCKER_PASS" psql \
    -h "$DOCKER_HOST" -p "$DOCKER_PORT" -U "$DOCKER_USER" \
    -c "CREATE DATABASE $DOCKER_DB;" postgres
echo "    Database '$DOCKER_DB' created."

# ── Create schema ──────────────────────────────────────────────────────────────
echo ""
echo "[3] Creating schema (21 tables) ..."
PGPASSWORD="$DOCKER_PASS" psql \
    -h "$DOCKER_HOST" -p "$DOCKER_PORT" -U "$DOCKER_USER" -d "$DOCKER_DB" \
    -f "$DATA_DIR/schematext.sql"
echo "    Schema created."

# ── Load CSV files ─────────────────────────────────────────────────────────────
echo ""
echo "[4] Loading CSV data (this will take ~10–20 minutes) ..."

# Table load order: small lookup tables first, large tables last
TABLES=(
    "comp_cast_type"
    "company_type"
    "info_type"
    "kind_type"
    "link_type"
    "role_type"
    "keyword"
    "complete_cast"
    "movie_link"
    "aka_title"
    "company_name"
    "movie_info_idx"
    "movie_keyword"
    "movie_companies"
    "aka_name"
    "title"
    "char_name"
    "name"
    "person_info"
    "movie_info"
    "cast_info"
)

TOTAL=${#TABLES[@]}
i=0
for TABLE in "${TABLES[@]}"; do
    i=$((i + 1))
    CSV_FILE="$DATA_DIR/${TABLE}.csv"
    if [ ! -f "$CSV_FILE" ]; then
        echo "    [$i/$TOTAL] WARNING: $CSV_FILE not found, skipping."
        continue
    fi
    SIZE=$(du -sh "$CSV_FILE" 2>/dev/null | cut -f1)
    echo -n "    [$i/$TOTAL] $TABLE ($SIZE) ... "
    T_START=$(date +%s)

    PGPASSWORD="$DOCKER_PASS" psql \
        -h "$DOCKER_HOST" -p "$DOCKER_PORT" -U "$DOCKER_USER" -d "$DOCKER_DB" \
        -c "\COPY ${TABLE} FROM '${CSV_FILE}' CSV ESCAPE '\'" \
        2>/dev/null

    T_END=$(date +%s)
    echo "done ($(( T_END - T_START ))s)"
done

# ── Verify row counts ──────────────────────────────────────────────────────────
echo ""
echo "[5] Verifying row counts ..."
PGPASSWORD="$DOCKER_PASS" psql \
    -h "$DOCKER_HOST" -p "$DOCKER_PORT" -U "$DOCKER_USER" -d "$DOCKER_DB" \
    -c "
SELECT table_name,
       (xpath('/row/cnt/text()',
        query_to_xml(format('SELECT COUNT(*) AS cnt FROM %I', table_name),
        false, true, '')))[1]::text::int AS row_count
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY row_count DESC;
"

# ── Create indexes for JOB query performance ──────────────────────────────────
echo ""
echo "[6] Creating indexes for JOB benchmark queries ..."
PGPASSWORD="$DOCKER_PASS" psql \
    -h "$DOCKER_HOST" -p "$DOCKER_PORT" -U "$DOCKER_USER" -d "$DOCKER_DB" << 'SQL'

-- cast_info (largest table ~36M rows)
CREATE INDEX ci_movie_id_idx   ON cast_info(movie_id);
CREATE INDEX ci_person_id_idx  ON cast_info(person_id);
CREATE INDEX ci_role_id_idx    ON cast_info(role_id);

-- movie_info
CREATE INDEX mi_movie_id_idx       ON movie_info(movie_id);
CREATE INDEX mi_info_type_id_idx   ON movie_info(info_type_id);

-- movie_info_idx
CREATE INDEX mi_idx_movie_id_idx     ON movie_info_idx(movie_id);
CREATE INDEX mi_idx_info_type_id_idx ON movie_info_idx(info_type_id);

-- movie_keyword
CREATE INDEX mk_movie_id_idx    ON movie_keyword(movie_id);
CREATE INDEX mk_keyword_id_idx  ON movie_keyword(keyword_id);

-- movie_companies
CREATE INDEX mc_movie_id_idx         ON movie_companies(movie_id);
CREATE INDEX mc_company_id_idx       ON movie_companies(company_id);
CREATE INDEX mc_company_type_id_idx  ON movie_companies(company_type_id);

-- movie_link
CREATE INDEX ml_movie_id_idx        ON movie_link(movie_id);
CREATE INDEX ml_linked_movie_id_idx ON movie_link(linked_movie_id);

-- name
CREATE INDEX n_gender_idx ON name(gender);

-- aka_name
CREATE INDEX an_person_id_idx ON aka_name(person_id);

-- title
CREATE INDEX t_production_year_idx ON title(production_year);
CREATE INDEX t_kind_id_idx         ON title(kind_id);

-- company_name
CREATE INDEX cn_country_code_idx ON company_name(country_code);

-- complete_cast
CREATE INDEX cc_movie_id_idx     ON complete_cast(movie_id);
CREATE INDEX cc_subject_id_idx   ON complete_cast(subject_id);

SQL

echo "    Indexes created."

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  IMDB loaded successfully into Docker PostgreSQL!"
echo ""
echo "  Next steps:"
echo "    1. Download JOB queries (113 SQL files):"
echo "       python experiments/hsm_job_validation.py --download"
echo ""
echo "    2. Run full HSM validation:"
echo "       python experiments/hsm_job_validation.py --execute"
echo "═══════════════════════════════════════════════════════════"
