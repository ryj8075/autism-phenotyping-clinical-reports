#!/bin/bash
# Attention / phenotype faithfulness validation — run both tests in order

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Attention / phenotype faithfulness validation"
echo "============================================================"

echo ""
echo "[1/2] TEST 1 — Faithfulness / representativeness"
python3 faithfulness.py

echo ""
echo "[2/2] TEST 2 — Attention concentrates on phenotype-bearing content"
python3 concentration.py

echo ""
echo "============================================================"
echo "  Done."
echo "    outputs/  : faithfulness_results.json, faithfulness_per_report.csv, faithfulness_sweep.csv,"
echo "                concentration_enrichment.csv, concentration_auroc.json"
echo "    figures/  : faithfulness_gold_reconstruction, faithfulness_full_reconstruction,"
echo "                concentration_enrichment_forest, concentration_auroc  (.png + .pdf)"
echo "============================================================"
