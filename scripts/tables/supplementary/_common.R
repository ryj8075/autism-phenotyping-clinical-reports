# =====================================================================
# Shared setup for the supplementary-table scripts (TableS1.R … TableS9.R).
# Each TableSN.R sources this, then builds its one table (one XLSX per script).
#   Canonical = full silver, 489 reports / 19 domains.
# =====================================================================
rm(list = ls())
suppressMessages(library(jsonlite))

.script_dir <- function() {
  a <- commandArgs(FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f[1])) else getwd()
}
SCRIPT_DIR <- .script_dir()
REPO_ROOT <- normalizePath(file.path(SCRIPT_DIR, "..", "..", ".."), mustWork = FALSE)

ROOT <- Sys.getenv("REPORT_LLM_ROOT", REPO_ROOT)
TS_DIR <- Sys.getenv("REPORT_LLM_TABLE_SOURCE_DIR", file.path(REPO_ROOT, "table_source"))
OUT <- Sys.getenv("REPORT_LLM_SUPPLEMENTARY_TABLE_OUT_DIR", file.path(REPO_ROOT, "tables", "supplementary"))
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

# input sources — all copied into table_source/
P_LAYER <- file.path(TS_DIR, "layer_wise_results_489samples_epoch40_153stc_128tkn_epoch40_patience10_no_headings.csv")
P_ATT   <- TS_DIR
P_TOPK  <- TS_DIR
P_SEED  <- file.path(TS_DIR, "bootstrap_permutation_results.json")
P_STEP2 <- file.path(TS_DIR, "step2_gmm_vs_heavytail_results.json")
P_STEP6 <- file.path(TS_DIR, "step6_seed_stability_results.json")
P_PROF  <- file.path(TS_DIR, "controlled_individual_profiles.tsv")
P_SIL   <- file.path(TS_DIR, "silver_labels_26reports.jsonl")
P_GOLD  <- file.path(TS_DIR, "gold_labels_26reports.jsonl")
P_SENT  <- file.path(TS_DIR, "sentence_level_validation_results.json")

# 19-domain order + codes (data-driven)
meta    <- fromJSON(file.path(TS_DIR, "domain_vectors_meta_latest.json"))
doms    <- meta$domain_columns
ndom    <- length(doms)
codes   <- c(paste0("A", 1:7), paste0("B", 1:9), paste0("C", seq_len(ndom - 16)))
stopifnot(length(codes) == ndom)
code_of <- setNames(codes, doms)

# Reader-facing domain names. Table1_phenotype_ontology.xlsx is the source of truth,
# so any table a reviewer reads shows these strings instead of the pipeline column ids.
display_of <- c(
  social_emotional_reciprocity = "Social-emotional reciprocity",
  nonverbal_communication      = "Nonverbal communication",
  relationship_play            = "Social relationships",
  stereotyped_behavior         = "Stereotyped behavior",
  insistence_on_sameness       = "Insistence on sameness",
  restricted_interests         = "Restricted interests",
  sensory_processing           = "Sensory reactivity",
  externalizing                = "Externalizing behaviors",
  internalizing                = "Internalizing behaviors",
  language_skills              = "Language skills",
  physiological_function       = "Physiological function",
  adaptive_behavior            = "Adaptive behavior",
  intelligence_learning        = "Intellectual functioning and learning skills",
  executive_function           = "Executive function",
  motor_skills                 = "Motor skills",
  family_environment           = "Family environment",
  test_scores                  = "Test scores",
  other_general                = "Other/general",
  recommendations              = "Recommendations")
stopifnot(all(doms %in% names(display_of)))

group_of <- c(A = "Core ASD domains",
              B = "Associated and co-occurring features",
              C = "Report elements and other")

read_jsonl <- function(p) lapply(readLines(p, warn = FALSE),
  function(l) if (nzchar(trimws(l))) fromJSON(l, simplifyVector = FALSE) else NULL)

# locate _common.R relative to the calling script (so `source()` works from anywhere)
.tbl_dir <- function() {
  a <- commandArgs(FALSE); f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(f[1]) else OUT
}

# format: one .xlsx per table
suppressMessages(library(openxlsx))
write_table <- function(df, base) {
  p <- file.path(OUT, paste0(base, ".xlsx"))
  openxlsx::write.xlsx(df, p, overwrite = TRUE)
  cat("  ->", basename(p), "\n")
  invisible(p)
}
