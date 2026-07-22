#!/usr/bin/env Rscript
# Table S6 — construction sensitivity of the phenotype-space results.
rm(list = ls())
source("_common.R")

# sentence budget (top-K)
sumf <- list.files(P_TOPK, pattern = "^topk_summary_.*\\.json$", full.names = TRUE)
stopifnot(length(sumf) > 0)
tk <- fromJSON(sort(sumf, decreasing = TRUE)[1])$summaries
tk <- tk[order(tk$K), ]
budget <- data.frame(
  construction_choice = "Sentence budget",
  setting        = ifelse(tk$K == 10, sprintf("K = %d (main)", tk$K), sprintf("K = %d", tk$K)),
  domains_used   = "19",
  eff_dim        = sprintf("%.2f", tk$effective_dim),
  pc1_variance   = sprintf("%.3f", tk$pc1_variance),
  subtype_BIC_winner = ifelse(tk$gmm_vs_t_winner == "t_dist", "t",
                              sprintf("GMM k%d", tk$best_gmm_k)),
  cross_seed_ARI_k2 = sprintf("%.3f", tk$cross_seed_ari_k2),
  lead_subspace_Mardia_z     = sprintf("%.2f", tk$lead3_z),
  residual_subspace_Mardia_z = sprintf("%.2f", tk$resid_z),
  heavy_tail_in_residual = ifelse(tk$pattern_preserved, "yes", "no"),
  stringsAsFactors = FALSE)

# domain set (ablation)
drp <- fromJSON(file.path(TS_DIR, "sensitivity_drop_domains_results.json"))$variants
dwn <- fromJSON(file.path(TS_DIR, "sensitivity_downweight_results.json"))$variants
hvy <- fromJSON(file.path(TS_DIR, "sensitivity_heavytail_results.json"))$variants

variants <- list(
  baseline_full19          = list(lab = "Full 19 domains (baseline)",     src = "drp"),
  drop_sentence_f1_lt_0.3  = list(lab = "Exclude sentence-F1 < 0.3",      src = "drp"),
  drop_report_rho_lt_0.3   = list(lab = "Exclude report-Spearman < 0.3",  src = "drp"),
  drop_report_rho_lt_0     = list(lab = "Exclude report-Spearman < 0",    src = "drp"),
  downweight_sentence_f1   = list(lab = "Down-weight by sentence-F1",     src = "dwn"),
  downweight_report_rho    = list(lab = "Down-weight by report-Spearman", src = "dwn"))
h12 <- list(drp = drp, dwn = dwn)

domset <- do.call(rbind, lapply(names(variants), function(k) {
  m <- h12[[variants[[k]]$src]][[k]]; h <- hvy[[k]]
  data.frame(
    construction_choice = "Domain set",
    setting        = variants[[k]]$lab,
    # down-weighting drives some domains to zero weight, so the count that survives is
    # what the run reports, not the nominal 19
    domains_used   = if (variants[[k]]$src == "dwn") sprintf("%d (weighted)", m$n_domains)
                     else as.character(m$n_domains),
    eff_dim        = sprintf("%.2f", m$eff_dim),
    pc1_variance   = sprintf("%.3f", m$pc1_evr),
    subtype_BIC_winner = if (isTRUE(m$t_wins)) "t" else sprintf("GMM k%d", m$best_gmm_k),
    cross_seed_ARI_k2 = sprintf("%.3f", m$cross_seed_ari_k2),
    lead_subspace_Mardia_z     = sprintf("%.2f", h$lead3_z),
    residual_subspace_Mardia_z = sprintf("%.2f", h$resid_z),
    heavy_tail_in_residual = ifelse(isTRUE(h$pattern_preserved), "yes", "no"),
    stringsAsFactors = FALSE)
}))

s6 <- rbind(budget, domset)
write_table(s6, "TableS6")
