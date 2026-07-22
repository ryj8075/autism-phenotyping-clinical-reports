#!/bin/bash
set -uo pipefail

EXPERIMENT_NAME=""
TOKENIZED_PATH=""
REPORT_MAX_LENGTH=153
SENTENCE_MAX_LENGTH=128
COLORBAR="fixed"
USE_SAVED=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_BASE="$SCRIPT_DIR/analysis_results"

if [ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "${CONDA_ENV:-report_llm}"
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_name) EXPERIMENT_NAME="$2"; shift 2;;
        --tokenized_path) TOKENIZED_PATH="$2"; shift 2;;
        --report_max_length) REPORT_MAX_LENGTH="$2"; shift 2;;
        --sentence_max_length) SENTENCE_MAX_LENGTH="$2"; shift 2;;
        --colorbar) COLORBAR="$2"; shift 2;;
        --reinfer) USE_SAVED=false; shift;;
        --help)
            echo "Usage: $0 --experiment_name <exp> --tokenized_path <dir> [options]"
            echo "  --experiment_name     Experiment name and trained model prefix [required]"
            echo "  --tokenized_path      Tokenized tensor directory [required]"
            echo "  --report_max_length   Maximum sentences per report [default: 153]"
            echo "  --sentence_max_length Maximum tokens per sentence [default: 128]"
            echo "  --colorbar            Heatmap colorbar: fixed | datarange [default: fixed]"
            echo "  --reinfer             Recompute out-of-sample inference from fold models"
            exit 0;;
        *) echo "[ERROR] Unknown option: $1"; exit 1;;
    esac
done

if [ -z "$EXPERIMENT_NAME" ] || [ -z "$TOKENIZED_PATH" ]; then
    echo "[ERROR] --experiment_name and --tokenized_path are required. See --help."; exit 1
fi

OUTPUT_DIR="$OUTPUT_BASE/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$ROOT_DIR/trained_models/${EXPERIMENT_NAME}_fold1.pt" ]; then
    echo "[ERROR] Missing model: $ROOT_DIR/trained_models/${EXPERIMENT_NAME}_fold1.pt"
    echo "        Run training first with the relevant run_train_*.sh script."
    exit 1
fi

echo "============================================================"
echo " Clinical report analysis"
echo "   experiment    : $EXPERIMENT_NAME"
echo "   tokenized_dir : $TOKENIZED_PATH"
echo "   settings      : ${REPORT_MAX_LENGTH}stc x ${SENTENCE_MAX_LENGTH}tkn, colorbar=$COLORBAR"
echo "   output        : $OUTPUT_DIR"
echo "============================================================"

cd "$SCRIPT_DIR"

if [ -f "$ROOT_DIR/intermediates/$EXPERIMENT_NAME/predicted_labels_np.npy" ] || \
   [ -f "$ROOT_DIR/intermediates/$EXPERIMENT_NAME/probs_np.npy" ]; then
    echo ""
    echo "[1/2] Confusion matrix"
    python confusion_matrix.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --output_dir "$OUTPUT_DIR"
else
    echo ""
    echo "[1/2] Confusion matrix skipped: no intermediate predictions"
fi

echo ""
echo "[2/2] Phenotype interpretation analysis"
PHENO_ARGS=(
    --experiment_name "$EXPERIMENT_NAME"
    --output_dir "$OUTPUT_DIR"
    --tokenized_path "$TOKENIZED_PATH"
    --report_max_length "$REPORT_MAX_LENGTH"
    --sentence_max_length "$SENTENCE_MAX_LENGTH"
    --colorbar "$COLORBAR"
)
if [ "$USE_SAVED" = true ]; then
    PHENO_ARGS+=(--use_saved_intermediates)
fi
python analyze_phenotypes_ko.py "${PHENO_ARGS[@]}"

echo ""
echo "============================================================"
echo " Analysis complete -> $OUTPUT_DIR"
echo "============================================================"
ls -la "$OUTPUT_DIR"
