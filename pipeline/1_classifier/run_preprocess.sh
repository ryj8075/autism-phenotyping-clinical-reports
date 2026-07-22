#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/preprocess_data_ko.py" \
  --input_type txt \
  --txt_dir "${TXT_DIR:-$SCRIPT_DIR/data/reports_txt}" \
  --meta_path "${META_PATH:-$SCRIPT_DIR/data/metadata/metadata.csv}" \
  --model_name "${MODEL_NAME:-klue/roberta-base}" \
  --output_dir "${OUTPUT_DIR:-$SCRIPT_DIR/data/reports_tokenized}" \
  --report_max_length "${REPORT_MAX_LENGTH:-153}" \
  --sentence_max_length "${SENTENCE_MAX_LENGTH:-128}"
