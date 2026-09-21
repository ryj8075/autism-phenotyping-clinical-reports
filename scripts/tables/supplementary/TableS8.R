#!/usr/bin/env Rscript
# Table S8 — patient-level prototypicality validation: silver vs gold Mahalanobis per report.
rm(list = ls())
source("_common.R")

RJ <- file.path(TS_DIR, "results_top10_vs_top10.json")
stopifnot(file.exists(RJ))
d <- fromJSON(RJ)
pr <- d$per_report                                   # 26 gold reports, ten-sentence vectors

s8 <- data.frame(report_id = pr$report_id,
                 cosine             = round(pr$cosine, 3),
                 silver_mahalanobis = round(pr$silver_mahalanobis_adj_loo, 3),
                 gold_mahalanobis   = round(pr$gold_mahalanobis_adj_loo, 3),
                 stringsAsFactors = FALSE)
s8 <- s8[order(-s8$gold_mahalanobis), ]              # most atypical (by gold) first
write_table(s8, "TableS8")
