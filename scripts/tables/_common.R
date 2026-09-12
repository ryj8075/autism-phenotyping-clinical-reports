#!/usr/bin/env Rscript
# =====================================================================
# Shared setup for the MAIN table scripts.
# Each TableN.R sources this, then writes one XLSX.
#   Canonical = full silver, 489 reports / 19 domains.
# =====================================================================
rm(list = ls())

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

META   <- file.path(TS_DIR, "metadata_489reports.csv")                # Table 1 cohort
RES    <- TS_DIR                                                      # Table 2 three selector txt files here

# format: one .xlsx per table
suppressMessages(library(openxlsx))
write_table <- function(df, base) {
  p <- file.path(OUT, paste0(base, ".xlsx"))
  openxlsx::write.xlsx(df, p, overwrite = TRUE)
  cat("  ->", basename(p), "\n")
  invisible(p)
}
