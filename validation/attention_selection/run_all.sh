#!/bin/bash
# ===========================================================================
# 1.7 Attention / phenotype faithfulness validation — run both tests in order
# ===========================================================================
# All input/output paths are centralised in _common.py (PATHS). No timestamps;
# fixed output filenames. Only standard libs (numpy/pandas/scipy/sklearn/
# matplotlib/pyyaml). This sub-pipeline only READS existing artifacts.
#
# Usage:
#   bash run_all.sh
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  1.7 Attention / phenotype faithfulness validation"
echo "============================================================"

echo ""
echo "[1/2] TEST 1 — Faithfulness / representativeness"
python3 test1_faithfulness.py

echo ""
echo "[2/2] TEST 2 — Attention concentrates on phenotype-bearing content"
python3 test2_concentration.py

echo ""
echo "============================================================"
echo "  Done."
echo "    outputs/  : test1_results.json, test1_per_report.csv, test1_sweep.csv,"
echo "                test2a_enrichment.csv, test2b_auroc.json"
echo "    figures/  : test1_gold_reconstruction, test1_full_reconstruction,"
echo "                test2a_enrichment_forest, test2b_auroc  (.png + .pdf)"
echo "============================================================"
