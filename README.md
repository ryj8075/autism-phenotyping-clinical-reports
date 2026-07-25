# autism-phenotyping-clinical-reports

Code for *"A transparent language-model pipeline for multidomain autism phenotyping from Korean clinical reports."*

This repository contains the analysis code that turns standardized Korean clinical reports into 19-domain phenotype vectors and runs the phenotype-space analyses reported in the paper. It does not contain clinical text or any patient-level data. See [Data](#data) below.

## Overview

The pipeline has four stages.

1. **Attention-based sentence selection.** A KLUE-RoBERTa classifier is fine-tuned for binary autism vs. non-autism prediction with stratified five-fold cross-validation, grouped on the individual identifier so that no individual appears in both the training and held-out folds. Sentence-level attention from the held-out folds ranks the sentences of each report.
2. **Silver labeling.** A local large language model (Llama-3.1-8B served through Ollama) assigns each sentence to one or more of 19 phenotype domains under a fixed prompt. Labels are aggregated over five queries per sentence and validated against expert gold annotations on a held-out set.
3. **Multidomain vectors.** The highest-attention sentences of each report are extracted, and each report becomes a domain-frequency vector over the 19 domains.
4. **Phenotype-space analysis.** The vectors are analyzed as compositional data in log-ratio coordinates, tested for discrete subtype structure, and summarized as a two-layer fingerprint of prototypicality and normative deviation.

## Repository structure

| Path | Contents |
| --- | --- |
| `ontology/` | The 19-domain ontology (`domains_19.yaml`: codes, groups, definitions) and the taxonomy builder |
| `pipeline/1_classifier/` | KLUE-RoBERTa fine-tuning, five-fold cross-validation, and sentence-attention extraction |
| `pipeline/2_silver_labeling/` | Sentence segmentation, local-LLM domain labeling, and gold-set evaluation |
| `pipeline/3_multidomain_vectors/` | Top-k attention-sentence extraction and domain-frequency vector construction |
| `pipeline/4_phenotype_space/` | Log-ratio transforms, subtype tests, and the two-layer fingerprint (step 1–11, plus the report-type confound check) |
| `validation/` | Attention faithfulness, silver-label agreement, vector robustness, and score-space checks |
| `scripts/figures/` | R scripts that generate the manuscript and supplementary figures |
| `scripts/tables/` | R scripts that generate the manuscript and supplementary tables |

## Requirements

- Python 3.10 (pipeline and validation)
- R with `ggplot2`, `patchwork`, and `jsonlite` (figures and tables)
- Ollama, for local LLM inference in the silver-labeling stage
- One CUDA GPU for classifier fine-tuning; the downstream analyses run on CPU

Create the Python environment with

```bash
conda env create -f pipeline/1_classifier/report_llm.yml
conda activate report_llm
```

## Reproducing the analyses

The stages are ordered and each consumes the output of the previous one. Input and output paths are set in each stage's `config.yaml` or in the shell script, and clinical text must be provided by the user (see [Data](#data)).

```bash
# Stage 1 — tokenize reports, fine-tune the classifier, extract sentence attention
bash pipeline/1_classifier/run_preprocess.sh
bash pipeline/1_classifier/fine_tuning/run_train_489reports.sh
bash pipeline/1_classifier/analysis/run_analysis.sh \
  --experiment_name <experiment> --tokenized_path <tokenized_dir>

# Stage 2 — segment sentences and assign domain labels
python pipeline/2_silver_labeling/run.py --config pipeline/2_silver_labeling/config.yaml

# Stage 3 — extract top-k sentences and build domain-frequency vectors
python pipeline/3_multidomain_vectors/top_10_sentences/extract_all_high_attention_sentences.py
python pipeline/3_multidomain_vectors/domain_frequency_vector/build_domain_vectors.py

# Stage 4 — run the phenotype-space analyses (confound check, then step 1–11)
bash pipeline/4_phenotype_space/run_all.sh
```

Validation analyses are independent and read existing stage outputs:

```bash
bash validation/attention_selection/run_all.sh              # attention faithfulness
bash validation/vector_robustness/run_topk_sensitivity.sh   # top-k robustness
# silver-label agreement and score-space checks: see the scripts under
# validation/silver_labels/ and validation/score_space/
```

Figures and tables are generated from the analysis outputs with the R scripts in `scripts/`:

```bash
Rscript scripts/figures/Figure2.R    # Figure3–6 and supplementary/FigureS1–6 likewise
Rscript scripts/tables/Table1.R      # Table2–3 and supplementary/TableS1–9 likewise
```

## Data

The clinical reports analyzed in this study contain potentially identifying and sensitive patient information and are not publicly available. De-identified derived data, including the domain-frequency phenotype vectors and aggregate analysis outputs, are available from the corresponding author on reasonable request, subject to approval by the Institutional Review Board of Kyung Hee University Hospital and completion of a data use agreement.

The pipeline expects report text and metadata under each stage's configured data directory, which is excluded from version control. Running the pipeline end to end therefore requires access to the source data. The stage-4 phenotype-space analyses can be run from the derived domain-frequency vectors alone.

## Citation

```bibtex
@article{TODO,
  title   = {A transparent language-model pipeline for multidomain autism phenotyping from Korean clinical reports},
  author  = {Ryu, Yeojin and An, Joon-Yong and Oh, Miae},
  year    = {TBD},
  journal = {TBD}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text. If you use this code in your own work, please cite the paper above.

## Contact

Correspondence to Miae Oh (miae612@khu.ac.kr).
