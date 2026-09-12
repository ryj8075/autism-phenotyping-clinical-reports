#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/preprocess_data_ko.py" \
  --input_type txt \
  --txt_dir "${TXT_DIR:-$SCRIPT_DIR/data/489reports/reports_txt_no_headings}" \
  --meta_path "${META_PATH:-$SCRIPT_DIR/data/metadata/metadata_489reports.csv}" \
  --model_name "${MODEL_NAME:-klue/roberta-base}" \
  --output_dir "${OUTPUT_DIR:-$SCRIPT_DIR/data/489reports/reports_tokenized_153stc_128tkn_no_headings}" \
  --report_max_length "${REPORT_MAX_LENGTH:-153}" \
  --sentence_max_length "${SENTENCE_MAX_LENGTH:-128}"
