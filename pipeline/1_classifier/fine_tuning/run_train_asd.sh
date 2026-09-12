#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-report_llm}"
if [[ -n "$CONDA_SH" && -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "$SCRIPT_DIR/roberta_sentence_attention_run_ko.py" \
  --experiment_name asd233samples_epoch40_153stc_128tkn_epoch40_patience10_no_headings \
  --tokenized_path "${TOKENIZED_PATH:-$ROOT_DIR/data/asd233/reports_tokenized_153stc_128tkn_no_headings}" \
  --epochs 40 \
  --patience 10 \
  --batch_size 1 \
  --lr 1e-5 \
  --save_fold_models \
  --save_intermediates \
  --no_wandb \
  --report_max_length 153 \
  --sentence_max_length 128
