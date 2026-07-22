#!/usr/bin/env Rscript
# Table S8 — patient-level prototypicality validation: silver vs gold Mahalanobis per report.
rm(list = ls())
source("_common.R")

RJ <- file.path(TS_DIR, "results_full_vs_gold_llama_full489_26gold.json")
stopifnot(file.exists(RJ))
d <- fromJSON(RJ)
pr <- d$per_report_mahalanobis                       # 26 gold reports

s9 <- data.frame(report_id = pr$report_id,
                 silver_mahalanobis = round(pr$silver_mahal_res, 3),
                 gold_mahalanobis   = round(pr$gold_mahal_res, 3),
                 stringsAsFactors = FALSE)
s9 <- s9[order(-s9$gold_mahalanobis), ]              # most atypical (by gold) first
write_table(s9, "TableS8")
