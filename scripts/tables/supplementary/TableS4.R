#!/usr/bin/env Rscript
# Table S4 — pseudocount sensitivity of the residual heavy-tail.
rm(list = ls())
source("_common.R")

J <- file.path(TS_DIR, "step1_5_pseudocount_sensitivity_results.json")
stopifnot(file.exists(J))
d <- fromJSON(J)
r <- d$results                                   # data.frame: one row per pseudocount

s5 <- data.frame(
  pseudocount           = format(r$pseudocount, scientific = TRUE),
  pc1_3_mardia_z        = round(r$pc1_3_z, 2),
  pc4plus_mardia_z      = round(r$pc4plus_z, 2),
  full_mardia_z         = round(r$full_z, 2),
  tail_pct_beyond_chi99 = round(100 * r$tail_frac_beyond_chi99, 1),
  check.names = FALSE, stringsAsFactors = FALSE)

write_table(s5, "TableS4")
