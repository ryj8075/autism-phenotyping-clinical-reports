#!/usr/bin/env Rscript
# Table 3 — sentence-selector (KLUE-RoBERTa) 5-fold CV performance, three independent models.
# All (n=489), A-type / autism-diagnostic (n=233), P-type / psych-assessment (n=256).
rm(list = ls())
source("_common.R")
files <- c(All = "489samples", A = "asd233samples", P = "psy256samples")
suffix <- "_epoch40_153stc_128tkn_epoch40_patience10_no_headings.txt"
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

# parse overall "METRIC: mean (+/- sd)"
parse_overall <- function(tag) {
  p <- file.path(RES, paste0(files[[tag]], suffix)); stopifnot(file.exists(p))
  L <- readLines(p, warn = FALSE)
  overall <- list()
  for (l in L) {
    mm <- regmatches(l, regexec("^([A-Za-z0-9_ ]+?):\\s+([0-9.]+)\\s*\\(\\+/-\\s*([0-9.]+)\\)", l))[[1]]
    if (length(mm) == 4) overall[[trimws(mm[2])]] <- sprintf("%.3f (%.3f)", as.numeric(mm[3]), as.numeric(mm[4]))
  }
  overall
}
P <- lapply(names(files), parse_overall); names(P) <- names(files)

# overall metrics: metric x {All, A, P}
metrics <- unique(unlist(lapply(P, names)))
metrics <- setdiff(metrics, "Loss")   # drop loss row
ta <- data.frame(Metric = metrics,
                 All = vapply(metrics, function(m) P$All[[m]] %||% NA_character_, character(1)),
                 `A-type` = vapply(metrics, function(m) P$A[[m]]   %||% NA_character_, character(1)),
                 `P-type` = vapply(metrics, function(m) P$P[[m]]   %||% NA_character_, character(1)),
                 check.names = FALSE,
                 stringsAsFactors = FALSE, row.names = NULL)
write_table(ta, "Table3")
