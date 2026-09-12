#!/usr/bin/env Rscript
# Table S5 — GMM vs multivariate-t model selection by k.
rm(list = ls())
source("_common.R")
stopifnot(file.exists(P_STEP2), file.exists(P_STEP6))

s2 <- fromJSON(P_STEP2); seed6 <- fromJSON(P_STEP6)
ks <- sort(as.integer(names(s2$per_k)))
s6 <- do.call(rbind, lapply(ks, function(k) {
  v <- s2$per_k[[as.character(k)]]
  data.frame(k = k, GMM_BIC = round(v$gmm_bic, 1), t_BIC = round(s2$t_dist$bic, 1),
             t_nu = round(s2$t_dist$nu, 1),
             dBIC_vs_t = round(v$dbic_vs_t, 1),
             silhouette = round(seed6$per_k[[as.character(k)]]$silhouette, 3),
             stringsAsFactors = FALSE)
}))
write_table(s6, "TableS5")
