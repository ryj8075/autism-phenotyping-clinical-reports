#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-${REPO_ROOT}/pipeline}"
NLP_ROOT="${NLP_ROOT:-${PIPELINE_ROOT}/1_classifier}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

EXPERIMENT="${EXPERIMENT:-489samples_epoch40_153stc_128tkn_epoch40_patience10_no_headings}"

case "${EXPERIMENT}" in
  489samples*)    DATA_SUBDIR="489reports" ;;
  asd233samples*) DATA_SUBDIR="asd233" ;;
  psy256samples*) DATA_SUBDIR="psy256" ;;
  *)              DATA_SUBDIR="" ;;
esac

ATTENTION_MATRIX="${ATTENTION_MATRIX:-${NLP_ROOT}/intermediates/${EXPERIMENT}/attention_matrices_np.npy}"
if [[ -n "${DATA_SUBDIR}" ]]; then
  TOKENIZED_DIR="${TOKENIZED_DIR:-${NLP_ROOT}/data/${DATA_SUBDIR}/reports_tokenized_153stc_128tkn_no_headings}"
else
  TOKENIZED_DIR="${TOKENIZED_DIR:-}"
fi

SILVER_LABELS="${SILVER_LABELS:-${PIPELINE_ROOT}/2_silver_labeling/data/silver_label/silver_labels_489reports.jsonl}"
DOMAINS_YAML="${DOMAINS_YAML:-${REPO_ROOT}/ontology/domains_19.yaml}"

K_VALUES="${K_VALUES:-5 10 15 20}"
N_PERM="${N_PERM:-1000}"
SEED="${SEED:-42}"

OUTPUT_DIR="${SCRIPT_DIR}/results"
FIGURE_DIR="${SCRIPT_DIR}/figures"
mkdir -p "${OUTPUT_DIR}" "${FIGURE_DIR}"

if [[ -z "${TOKENIZED_DIR}" ]]; then
  echo "[ERROR] Could not infer TOKENIZED_DIR for EXPERIMENT='${EXPERIMENT}'. Set TOKENIZED_DIR explicitly." >&2
  exit 1
fi
for f in "${ATTENTION_MATRIX}" "${SILVER_LABELS}" "${DOMAINS_YAML}"; do
  [[ -e "${f}" ]] || { echo "[ERROR] Missing file: ${f}" >&2; exit 1; }
done
[[ -d "${TOKENIZED_DIR}" ]] || { echo "[ERROR] Missing directory: ${TOKENIZED_DIR}" >&2; exit 1; }

echo "[1] EXPERIMENT:       ${EXPERIMENT}"
echo "[1] ATTENTION_MATRIX: ${ATTENTION_MATRIX}"
echo "[1] TOKENIZED_DIR:    ${TOKENIZED_DIR}"
echo "[1] SILVER_LABELS:    ${SILVER_LABELS}"
echo "[1] DOMAINS_YAML:     ${DOMAINS_YAML}"
echo "[1] K_VALUES:         ${K_VALUES}"
echo ""

"${PYTHON_BIN}" "${SCRIPT_DIR}/topk_sensitivity.py" \
    --attention_matrix "${ATTENTION_MATRIX}" \
    --tokenized_dir "${TOKENIZED_DIR}" \
    --silver_labels "${SILVER_LABELS}" \
    --domains_yaml "${DOMAINS_YAML}" \
    --k_values ${K_VALUES} \
    --n_perm "${N_PERM}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}"
