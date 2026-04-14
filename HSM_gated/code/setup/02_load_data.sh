#!/bin/bash
# =============================================================
# TPC-H Data Loader for PostgreSQL
# HSM Throughput Experiments
#
# Usage:
#   bash 02_load_data.sh <scale_factor> [db_name] [pg_user]
#
# Examples:
#   bash 02_load_data.sh 0.2
#   bash 02_load_data.sh 1
#   bash 02_load_data.sh 3
#   bash 02_load_data.sh 10
#
# Prerequisites:
#   1. PostgreSQL must be running
#   2. Schema must be created (run 01_create_tables.sql first)
#   3. dbgen must be compiled in TPC-H V3.0.1/dbgen/
#   4. psql must be in PATH
# =============================================================

set -e  # Exit on error

# ---- Configuration ----
SF=${1:-"1"}          # Scale Factor
DB=${2:-"tpch"}       # Database name
USER=${3:-"postgres"} # PostgreSQL user

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DBGEN_DIR="$BASE_DIR/TPC-H V3.0.1/dbgen"
DATA_DIR="$BASE_DIR/data/sf${SF}"

echo "============================================"
echo " TPC-H Data Loader"
echo " Scale Factor : SF = $SF  (~$(echo "$SF * 1" | awk '{printf "%.0f", $1}') GB)"
echo " Database     : $DB"
echo " User         : $USER"
echo " Data Dir     : $DATA_DIR"
echo "============================================"

# ---- Step 1: Create database if not exists ----
echo "[1/4] Creating database '$DB' if not exists..."
psql -U "$USER" -tc "SELECT 1 FROM pg_database WHERE datname = '$DB'" | grep -q 1 \
    || psql -U "$USER" -c "CREATE DATABASE $DB;"

# ---- Step 2: Apply schema ----
echo "[2/4] Applying schema..."
psql -U "$USER" -d "$DB" -f "$SCRIPT_DIR/01_create_tables.sql"

# ---- Step 3: Check that .tbl files exist ----
echo "[3/4] Checking TPC-H data files (SF=$SF)..."

if ! ls "$DATA_DIR"/*.tbl 1>/dev/null 2>&1; then
    echo ""
    echo "  ERROR: No .tbl files found in $DATA_DIR"
    echo "  Please run this first:"
    echo "    bash setup/00_compile_and_generate.sh $SF"
    exit 1
fi
echo "  Found $(ls "$DATA_DIR"/*.tbl | wc -l) .tbl files in $DATA_DIR"

# ---- Step 4: Load .tbl files into PostgreSQL ----
# TPC-H .tbl files use '|' as delimiter with trailing '|'
echo "[4/4] Loading data into PostgreSQL..."

TABLES=("region" "nation" "supplier" "part" "partsupp" "customer" "orders" "lineitem")

for TABLE in "${TABLES[@]}"; do
    TBL_FILE="$DATA_DIR/${TABLE}.tbl"
    if [ -f "$TBL_FILE" ]; then
        echo "  Loading $TABLE..."
        # Remove trailing '|' from each line before loading
        sed 's/|$//' "$TBL_FILE" | psql -U "$USER" -d "$DB" \
            -c "COPY $TABLE FROM STDIN WITH (FORMAT CSV, DELIMITER '|');"
    else
        echo "  WARNING: $TBL_FILE not found — skipping"
    fi
done

# Run ANALYZE for query planner
echo "  Running ANALYZE..."
psql -U "$USER" -d "$DB" -c "ANALYZE;"

echo ""
echo "============================================"
echo " Data loading complete for SF=$SF"
echo " Database: $DB"
echo "============================================"
