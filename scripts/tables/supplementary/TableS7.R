#!/usr/bin/env Rscript
# Table S7 — cluster stability per k: cross-seed ARI (init) + bootstrap ARI (sample) + permutation null.
rm(list = ls())
source("_common.R")
stopifnot(file.exists(P_STEP6), file.exists(P_SEED))

cm <- fromJSON(P_STEP6); bp <- fromJSON(P_SEED)   # cm = step6 seed-stability (seed_ari per k)
ks <- sort(as.integer(names(cm$per_k)))

s5 <- do.call(rbind, lapply(ks, function(k) {
  v  <- cm$per_k[[as.character(k)]]
  bk <- bp$per_k[[as.character(k)]]          # bootstrap/perm for this k (NULL if not computed)
  hb <- !is.null(bk)
  data.frame(k = k,
             cross_seed_ARI_mean = round(v$seed_ari_mean, 3),
             cross_seed_ARI_sd   = round(v$seed_ari_sd, 3),
             # 0.8 is the reproducibility bar the manuscript and the Figure 5 legend use
             stable_ARI_ge_0.8   = ifelse(v$seed_ari_mean >= 0.8, "yes", "no"),
             bootstrap_ARI_mean  = if (hb) round(bk$bootstrap$mean, 3) else NA_real_,
             bootstrap_ARI_sd    = if (hb) round(bk$bootstrap$std, 3)  else NA_real_,
             permutation_null_mean    = if (hb) signif(bk$null$mean, 2) else NA_real_,
             permutation_null_95ci_hi = if (hb) round(bk$null$ci95_hi, 3) else NA_real_,
             perm_p_one_sided = if (hb) signif(bk$effect_size$p_value_one_sided, 2) else NA_real_,
             cohens_d_vs_null = if (hb) round(bk$effect_size$cohens_d_bootstrap_vs_null, 2) else NA_real_,
             stringsAsFactors = FALSE)
}))
write_table(s5, "TableS7")
