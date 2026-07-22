#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"

fail=0
run () {
  echo "> $1"
  if python3 "$1"; then echo "  done"; else echo "  FAIL: $1"; fail=1; fi
}

run confound_reanalysis.py      # all 489 reports
run report_type_figure_data.py  # 346 ASD reports, data for Supplementary Figure S4

[ $fail -eq 0 ] && echo "confound check completed" || echo "confound check had failures"
