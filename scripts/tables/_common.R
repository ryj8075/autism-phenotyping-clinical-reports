#!/usr/bin/env Rscript
# =====================================================================
# Shared setup for the MAIN table scripts.
# Each TableN.R sources this, then writes one XLSX.
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
REPO_ROOT <- normalizePath(file.path(SCRIPT_DIR, "..", ".."), mustWork = FALSE)

TS_DIR <- Sys.getenv("REPORT_LLM_TABLE_SOURCE_DIR", file.path(REPO_ROOT, "table_source"))
OUT <- Sys.getenv("REPORT_LLM_TABLE_OUT_DIR", file.path(REPO_ROOT, "tables"))
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

META   <- file.path(TS_DIR, "metadata_489reports.csv")                # Table 2 (cohort)
RES    <- TS_DIR                                                      # Table 3 reads the three selector *.txt here
TSV    <- file.path(TS_DIR, "domain_vectors_proportion_latest.tsv")   # Table 4 (prototype centroid)

# 19-domain order + codes (data-driven)
meta    <- fromJSON(file.path(TS_DIR, "domain_vectors_meta_latest.json"))
doms    <- meta$domain_columns
ndom    <- length(doms)
codes   <- c(paste0("A", 1:7), paste0("B", 1:9), paste0("C", seq_len(ndom - 16)))
stopifnot(length(codes) == ndom)
code_of <- setNames(codes, doms)

# format: one .xlsx per table
suppressMessages(library(openxlsx))
write_table <- function(df, base) {
  p <- file.path(OUT, paste0(base, ".xlsx"))
  openxlsx::write.xlsx(df, p, overwrite = TRUE)
  cat("  ->", basename(p), "\n")
  invisible(p)
}
