#!/bin/bash
# =============================================================
# TPC-H dbgen — Compile & Generate Data
# HSM Throughput Experiments
#
# Run this ONCE before anything else.
# Compiles dbgen from source, then generates .tbl files
# for all 4 scale factors: 0.2, 1.0, 3.0, 10.0
#
# Usage:
#   bash 00_compile_and_generate.sh          # all 4 SF
#   bash 00_compile_and_generate.sh 0.2      # single SF only
#   bash 00_compile_and_generate.sh 0.2 1.0  # specific SFs
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DBGEN_DIR="$BASE_DIR/TPC-H V3.0.1/dbgen"
DATA_BASE="$BASE_DIR/data"

# Scale factors to generate (can be overridden by args)
if [ "$#" -ge 1 ]; then
    SF_LIST=("$@")
else
    SF_LIST=("0.2" "1" "3" "10")
fi

# ── Step 1: Compile dbgen ──────────────────────────────────────────────────────
echo "============================================"
echo " Step 1: Compile TPC-H dbgen"
echo "============================================"

cd "$DBGEN_DIR"

if [ -f "./dbgen" ]; then
    echo "  dbgen already compiled — skipping compilation."
    echo "  (Delete ./dbgen to force recompile)"
else
    echo "  Configuring Makefile..."

    # Remove old Makefile and build artifacts from previous failed attempt
    rm -f Makefile Makefile.bak malloc.h *.o dbgen qgen
    cp makefile.suite Makefile

    # Detect OS and architecture
    # Detect OS — macOS uses LINUX machine type (POSIX-compatible, no MACOS in TPC-H)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS (Intel or Apple Silicon M1/M2/M3/M4)
        # Verify Xcode CLI tools are installed (provides gcc = Apple Clang)
        if ! command -v gcc &>/dev/null; then
            echo ""
            echo "  ERROR: Xcode Command Line Tools not found."
            echo "  Fix: Open Terminal and run:"
            echo "    xcode-select --install"
            exit 1
        fi
        echo "  Detected macOS (gcc = $(gcc --version | head -1))"

        # ── macOS fix: create malloc.h shim ──────────────────────────────────
        # macOS does NOT have a standalone <malloc.h>; it's part of <stdlib.h>
        # TPC-H bm_utils.c uses #include <malloc.h> → create a local shim file
        # that the compiler will find first (via -I. flag added to CFLAGS below)
        cat > malloc.h << 'MALLOC_SHIM'
/* malloc.h shim for macOS — Apple Clang uses stdlib.h instead */
#include <stdlib.h>
MALLOC_SHIM
        echo "  Created malloc.h shim for macOS compatibility"

        # macOS sed requires '' after -i for in-place edit
        sed -i '' \
            -e 's/^CC[[:space:]]*=[[:space:]]*$/CC      = gcc/' \
            -e 's/^DATABASE[[:space:]]*=[[:space:]]*$/DATABASE= SQLSERVER/' \
            -e 's/^MACHINE[[:space:]]*=[[:space:]]*$/MACHINE = LINUX/' \
            -e 's/^WORKLOAD[[:space:]]*=[[:space:]]*$/WORKLOAD = TPCH/' \
            Makefile

        # Add -I. so compiler finds local malloc.h shim first
        # Also suppress warnings from old C code on modern Clang
        sed -i '' \
            's/^CFLAGS[[:space:]]*=\(.*\)$/CFLAGS  =\1 -I. -Wno-implicit-function-declaration -Wno-return-type -Wno-unused-result/' \
            Makefile
    else
        # Linux
        echo "  Detected Linux"
        sed -i \
            -e 's/^CC[[:space:]]*=[[:space:]]*$/CC      = gcc/' \
            -e 's/^DATABASE[[:space:]]*=[[:space:]]*$/DATABASE= SQLSERVER/' \
            -e 's/^MACHINE[[:space:]]*=[[:space:]]*$/MACHINE = LINUX/' \
            -e 's/^WORKLOAD[[:space:]]*=[[:space:]]*$/WORKLOAD = TPCH/' \
            Makefile
    fi

    echo "  Compiling (OS=$OSTYPE)..."
    make -s dbgen 2>&1 | grep -v "^$" | tail -10

    if [ ! -f "./dbgen" ]; then
        echo ""
        echo "  ERROR: Compilation failed. Try manually:"
        echo "    cd \"$DBGEN_DIR\""
        echo "    nano Makefile   # set CC=gcc, DATABASE=SQLSERVER, MACHINE=LINUX, WORKLOAD=TPCH"
        echo "    make dbgen"
        exit 1
    fi

    echo "  ✓ dbgen compiled successfully"
fi

# ── Step 2: Generate .tbl files for each SF ───────────────────────────────────
echo ""
echo "============================================"
echo " Step 2: Generate TPC-H Data Files"
echo "============================================"

for SF in "${SF_LIST[@]}"; do
    # Normalize folder name: 0.2→sf0.2, 1→sf1, etc.
    SF_LABEL=$(echo "$SF" | sed 's/\./_/')
    OUT_DIR="$DATA_BASE/sf${SF}"
    mkdir -p "$OUT_DIR"

    # Check if already generated
    if ls "$OUT_DIR"/*.tbl 1>/dev/null 2>&1; then
        echo "  SF=$SF: .tbl files already exist in $OUT_DIR — skipping"
        echo "  (Delete $OUT_DIR/*.tbl to regenerate)"
        continue
    fi

    echo "  SF=$SF: Generating data..."
    echo "    Expected size: ~$(echo "$SF * 1024" | bc -l | xargs printf "%.0f") MB"

    START=$(date +%s)
    ./dbgen -s "$SF" -f -q   # -q = quiet mode
    END=$(date +%s)
    ELAPSED=$((END - START))

    # Move .tbl files to data folder
    mv *.tbl "$OUT_DIR/"

    TOTAL_SIZE=$(du -sh "$OUT_DIR" | cut -f1)
    echo "    ✓ Done in ${ELAPSED}s — Folder size: $TOTAL_SIZE"
    echo "    Files: $(ls "$OUT_DIR"/*.tbl | wc -l) tables"
done

echo ""
echo "============================================"
echo " All data generated successfully!"
echo ""
echo " Next steps:"
echo "   1. Set PostgreSQL credentials via env vars:"
echo "        cp .env.example .env    # at repo root"
echo "        edit .env (set HSM_DB_HOST/USER/PASSWORD)"
echo "        source .env"
echo ""
echo "   2. Load data into PostgreSQL:"
echo "      bash setup/02_load_data.sh 0.2"
echo "      bash setup/02_load_data.sh 1"
echo "      bash setup/02_load_data.sh 3"
echo "      bash setup/02_load_data.sh 10"
echo ""
echo "   3. Run experiments:"
echo "      python experiments/experiment_runner.py --quick"
echo "============================================"
