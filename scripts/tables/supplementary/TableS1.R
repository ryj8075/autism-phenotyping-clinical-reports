#!/usr/bin/env Rscript
# Table S1 — attention top-k faithfulness.
rm(list = ls())
source("_common.R")
stopifnot(file.exists(file.path(P_ATT, "test1_results.json")))

t1 <- fromJSON(file.path(P_ATT, "test1_results.json"), simplifyVector = FALSE)
Ks <- sort(as.integer(names(t1$gold_reference)))

# cosine_to_gold: agreement with the held-out expert-gold profile
# cosine_to_full: reconstruction of the report's own full-sentence 19-domain profile
mk <- function(blk, label) do.call(rbind, lapply(Ks, function(k) {
  co <- blk[[as.character(k)]]$cosine
  data.frame(K = k, reference = label,
             topK_median   = round(co$topK_median, 3),
             randomK_median = round(co$randK_median, 3),
             wilcoxon_p     = signif(co$topK_vs_randK$wilcoxon_p, 2),
             rank_biserial  = round(co$topK_vs_randK$effect_rank_biserial, 3),
             n              = co$topK_vs_randK$n,
             stringsAsFactors = FALSE)
}))
s2 <- rbind(mk(t1$full_reference, "full-report vector"),
            mk(t1$gold_reference, "expert-gold profile"))
write_table(s2, "TableS1")
