# autism-phenotyping-clinical-reports

Code for *"A transparent language-model pipeline for multidomain autism phenotyping from Korean clinical reports."*

This repository contains the analysis code that turns standardized Korean clinical reports into 19-domain phenotype vectors and runs the phenotype-space analyses reported in the paper. It does not contain clinical text or any patient-level data. See [Data](#data) below.

## Overview

The corpus is 489 structured narrative clinical reports from a single tertiary center, covering 372 individuals. 346 reports carry a confirmed autism diagnosis. Reports come from two documentation streams, autism diagnostic reports and psychological assessment reports.

The pipeline has four stages.

1. **Attention-based sentence selection.** A KLUE-RoBERTa classifier is fine-tuned for binary autism vs. non-autism prediction with stratified five-fold cross-validation, grouped on the individual identifier so that no individual appears in both the training and held-out folds. Sentence-level attention from the held-out folds ranks the sentences of each report.
2. **Silver labeling.** A local large language model, Llama-3.1-8B served through Ollama, assigns each sentence to one or more of 19 phenotype domains under a fixed prompt. Each sentence is queried five times. A domain enters the label when it appears in at least three of the five samples with a mean confidence of at least 0.5, capped at three domains per sentence. The labels are validated against expert gold annotations on a 26-report gold set.
3. **Multidomain vectors.** The ten highest-attention sentences of each report are extracted, and each report becomes a domain-frequency vector over the 19 domains.
4. **Phenotype-space analysis.** The vectors are analyzed as compositional data in log-ratio coordinates, residualized on report type, tested for discrete subtype structure, and summarized as a two-layer fingerprint of prototypicality and normative deviation.

The 19 domains fall into seven Core ASD domains, nine Associated and co-occurring features, and three Report elements and other. Codes, group assignments, and definitions live in `ontology/domains_19.yaml`.

## Repository structure

| Path | Contents |
| --- | --- |
| `ontology/` | The 19-domain ontology in `domains_19.yaml` and the taxonomy builder |
| `pipeline/1_classifier/` | KLUE-RoBERTa fine-tuning, five-fold cross-validation, and sentence-attention extraction |
| `pipeline/2_silver_labeling/` | Sentence segmentation, local-LLM domain labeling, gold-template creation, and gold-set evaluation |
| `pipeline/3_multidomain_vectors/` | Top-k attention-sentence extraction and domain-frequency vector construction |
| `pipeline/4_phenotype_space/` | Log-ratio transforms, subtype tests, and the two-layer fingerprint as step 1 through step 11, plus the report-type confound check |
| `validation/attention_selection/` | Whether attention-selected sentences reconstruct the report phenotype better than a random draw |
| `validation/silver_labels/` | Sentence-level and report-level agreement with gold, five-vote reliability, and domain-set sensitivity under `domain_set/` |
| `validation/vector_robustness/` | Sensitivity of the phenotype-space conclusions to the top-k sentence budget |
| `validation/score_space/` | Checks against standardized clinical scores on the subset that has them |
| `scripts/figures/` | R scripts that generate the manuscript and supplementary figures |
| `scripts/tables/` | R scripts that generate the manuscript and supplementary tables |

## Requirements

Two Python environments were used. Stages 1 and 2 ran on an on-premise GPU server and stages 3 and 4 ran on a local machine, so the two have different package sets.

**Stage 1, classifier fine-tuning.** One NVIDIA A100 80GB GPU, Python 3.10, torch, transformers, and scikit-learn. The environment file is `pipeline/1_classifier/report_llm.yml`.

```bash
conda env create -f pipeline/1_classifier/report_llm.yml
```

```bash
conda activate report_llm
```

**Stage 2, silver labeling.** Python 3.10 with the packages in `pipeline/2_silver_labeling/requirements.txt`, plus Ollama serving the labeling model on an on-premise GPU server. No report text leaves the machine.

**Stages 3 and 4 and the validation analyses.** Python 3.10 with NumPy, SciPy, pandas, scikit-learn, matplotlib, and PyYAML. These run on CPU. The top-k sentence extraction in stage 3 also needs torch and transformers to load the stage-1 tensors and the tokenizer, and the top-k sensitivity analysis needs torch. Both run on CPU as well.

**Figures and tables.** R with `ggplot2`, `patchwork`, `jsonlite`, `openxlsx`, `ragg`, `reshape2`, and `MASS`.

The exact versions behind the reported numbers are listed in the Methods section of the paper, separately for the training environment and the analysis environment.

## Reproducing the analyses

The stages are ordered and each consumes the output of the previous one. Input and output paths are set in each stage's `config.yaml` or in the shell script, and most also accept an environment variable. Clinical text must be provided by the user. See [Data](#data).

```bash
# Stage 1 — strip section headings, tokenize reports, fine-tune the classifier, extract sentence attention
python pipeline/1_classifier/remove_headings.py pipeline/1_classifier/data/489reports/reports_txt \
  -o pipeline/1_classifier/data/489reports/reports_txt_no_headings
bash pipeline/1_classifier/run_preprocess.sh
bash pipeline/1_classifier/fine_tuning/run_train_489reports.sh
bash pipeline/1_classifier/analysis/run_analysis.sh \
  --experiment_name <experiment> --tokenized_path <tokenized_dir>
```

The reported classifier was trained on report text with the `## … ##` section headings removed, which is what `remove_headings.py` does.

```bash
# Stage 2 — segment sentences and assign domain labels
python pipeline/2_silver_labeling/run.py --config pipeline/2_silver_labeling/config.yaml
```

```bash
# Stage 3 — extract top-k sentences and build domain-frequency vectors
python pipeline/3_multidomain_vectors/top_10_sentences/extract_all_high_attention_sentences.py
python pipeline/3_multidomain_vectors/domain_frequency_vector/build_domain_vectors.py
```

```bash
# Stage 4 — run the phenotype-space analyses
bash pipeline/4_phenotype_space/run_all.sh
```

Stage 4 runs `confound_check/` first and then step 1 through step 11. The confound check quantifies how much of the domain composition tracks report type and writes the data behind Supplementary Figure S4. Every step derives its own type-residual coordinates, so the ordering is a convention rather than a data dependency.

The validation analyses are independent of one another and read existing stage outputs.

```bash
bash validation/attention_selection/run_all.sh
```

```bash
bash validation/vector_robustness/run_topk_sensitivity.sh
```

The silver-label and score-space checks are run one script at a time. Under `validation/silver_labels/`, run `sentence_level.py` and `report_level.py` before anything in `domain_set/`, which reads their output.

### Figures and tables

Collect every output the figures need in `figure_source` and every output the tables need in `table_source`, then point the scripts at those two folders.

Most files keep the name their producer gave them. These are the exceptions.

| Produced as | Copy in as | Folder |
| --- | --- | --- |
| `confound_check/report_type_figure_data.json` | `report_type_confound.json` | `figure_source` |
| `attention_selection/outputs/faithfulness_results.json` | `attention_faithfulness_results.json` | `figure_source` |
| `attention_selection/outputs/faithfulness_results.json` | `faithfulness_results.json`, unchanged | `table_source` |
| `domain_set/drop_domains_robustness_results.json` | `sensitivity_drop_domains_results.json` | `table_source` |
| `domain_set/downweight_robustness_results.json` | `sensitivity_downweight_results.json` | `table_source` |
| `domain_set/heavytail_sensitivity_results.json` | `sensitivity_heavytail_results.json` | `table_source` |

Two inputs you prepare yourself, both for `figure_source`.

- `attention_exemplars_np.npy`, read by `FigureS2.R`. No script produces it. Cut a two-report slice from the stage-1 `attention_matrices_np.npy`, one A-type report and one P-type report.
- `figS1_embedding_pca.csv`, read by `FigureS1.R`. Run `pipeline/1_classifier/analysis/FigureS1_make_data.py` after stage 1, then copy what it writes to `pipeline/1_classifier/figure_source/`.

```bash
export REPORT_LLM_FIGURE_SOURCE_DIR=/path/to/figure_source
export REPORT_LLM_TABLE_SOURCE_DIR=/path/to/table_source
```

Each script sources its `_common.R` by relative path, so run it from its own directory.

```bash
cd scripts/figures && Rscript Figure2.R
```

```bash
cd scripts/figures/supplementary && Rscript FigureS1.R
```

```bash
cd scripts/tables && Rscript Table1.R
```

```bash
cd scripts/tables/supplementary && Rscript TableS1.R
```

Script names match the numbering in the paper.

### Paths and environment variables

Defaults assume the layout in this repository. Override them when your data sits elsewhere.

| Variable | Used by |
| --- | --- |
| `REPORT_TXT_DIR` | Raw report text, stage 1 only and only with `--data_mode txt` |
| `TOKENIZED_PATH` | Tokenized tensors, stage 1 training |
| `SILVER_LABELS_JSONL`, `GOLD_LABELS_JSONL` | Silver and gold label files |
| `DOMAIN_VECTOR_OUTPUT_DIR` | Stage 3 output directory, read by stage 4 |
| `DOMAIN_VECTORS_TSV`, `DOMAIN_META_JSON` | Individual stage 3 artifacts |
| `PHENOTYPE_SPACE_DIR` | Location of `_common_controlled.py` for the validation scripts |
| `SENTENCE_LEVEL_VALIDATION_JSON`, `REPORT_LEVEL_VALIDATION_JSON` | Gold-fidelity inputs for the domain-set sensitivity analyses |
| `REPORT_LLM_FIGURE_SOURCE_DIR`, `REPORT_LLM_TABLE_SOURCE_DIR` | Collected inputs for the R scripts |
| `REPORT_LLM_TABLE_OUT_DIR`, `REPORT_LLM_SUPPLEMENTARY_TABLE_OUT_DIR`, `REPORT_LLM_SUPPLEMENTARY_FIGURE_OUT_DIR` | Where the R scripts write |
| `CLASSIFIER_INTERMEDIATES_DIR`, `CLASSIFIER_METADATA_CSV` | Stage 1 intermediates and the report metadata CSV, read by `FigureS1_make_data.py` |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Stage 2, only when `config.yaml` selects the `openai` or `anthropic` provider instead of the local model |
| `SILVER_LABELING_DATA_DIR` | Folder holding the silver and gold label files, read by `validation/silver_labels/sentence_level.py` |
| `SILVER_VALIDATION_DIR` | Folder holding the sentence-level and report-level validation outputs, read by `domain_set/` |
| `REPORT_LLM_DATA_ROOT`, `PHENOTYPE_TABLE_LONG` | Standardized clinical score table for `validation/score_space/` |
| `DOMAIN_MATCHING_DETAILS_JSON` | Stage 3 sentence-to-label matching record, read by `validation/score_space/reconstruction_asymmetry.py` |

The stage-1 shell scripts and `validation/vector_robustness/run_topk_sensitivity.sh` read further variables, listed at the top of each script.

## Data

The clinical reports analyzed in this study contain potentially identifying and sensitive patient information and are not publicly available. De-identified derived data, including the domain-frequency phenotype vectors and aggregate analysis outputs, are available from the corresponding author on reasonable request, subject to approval by the Institutional Review Board of Kyung Hee University Hospital and completion of a data use agreement.

The pipeline expects report text and metadata under each stage's configured data directory, which is excluded from version control. Running the pipeline end to end therefore requires access to the source data. The stage-4 phenotype-space analyses can be run from the derived domain-frequency vectors alone.

Two label files carry the names the pipeline expects. `silver_labels_489reports.jsonl` holds the model-assigned labels for all 32,317 sentences of the 489-report corpus. `gold_labels_26reports.jsonl` holds the expert annotations for the 26-report gold set.

## Citation

```bibtex
@article{TODO,
  title   = {A transparent language-model pipeline for multidomain autism phenotyping from Korean clinical reports},
  author  = {Ryu, Yeojin and An, Joon-Yong and Yoo, Hee Jeong and Bzdok, Danilo and Mottron, Laurent and Oh, Miae},
  year    = {TBD},
  journal = {TBD}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text. If you use this code in your own work, please cite the paper above.

## Contact

Correspondence to Miae Oh (miae612@khu.ac.kr).
