#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"

echo "> t1_confound_reanalysis.py"
if python t1_confound_reanalysis.py; then echo "confound check completed"; else echo "FAIL"; fi
