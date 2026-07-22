#!/usr/bin/env Rscript
# Table S3 — silver-vs-gold agreement per domain.
rm(list = ls())
source("_common.R")
stopifnot(file.exists(P_SIL), file.exists(P_GOLD), file.exists(P_SENT))

# sentence-level PRF (analysis 2_1), keyed by domain
sent <- fromJSON(P_SENT)$per_domain
sf1  <- setNames(sent$f1, sent$domain)
spr  <- setNames(sent$precision, sent$domain)
src_ <- setNames(sent$recall, sent$domain)
ssup <- setNames(sent$support, sent$domain)

silver <- Filter(Negate(is.null), read_jsonl(P_SIL))
gold   <- Filter(Negate(is.null), read_jsonl(P_GOLD))
reps   <- sort(unique(vapply(gold, function(r) r$report_id, character(1))))

get_domains <- function(labels) {
  if (is.null(labels) || length(labels) == 0) return(character(0))
  vapply(labels, function(x) if (is.list(x)) as.character(x$domain_id) else as.character(x), character(1))
}
zero <- function() matrix(0, length(reps), ndom, dimnames = list(reps, doms))
Sc <- zero(); Gc <- zero()
for (r in silver) { rid <- r$report_id; if (rid %in% reps) for (d in get_domains(r$labels)) if (d %in% doms) Sc[rid, d] <- Sc[rid, d] + 1 }
for (r in gold)   { rid <- r$report_id; for (d in get_domains(r$labels)) if (d %in% doms) Gc[rid, d] <- Gc[rid, d] + 1 }
S <- Sc / pmax(rowSums(Sc), 1); G <- Gc / pmax(rowSums(Gc), 1)

tier <- function(rho) if (is.na(rho)) "weak" else if (rho >= 0.80) "strong" else if (rho >= 0.60) "moderate" else "weak"
s3 <- do.call(rbind, lapply(doms, function(d) {
  s <- S[, d]; g <- G[, d]; ok <- sd(s) > 1e-10 && sd(g) > 1e-10
  rho <- if (ok) suppressWarnings(cor(s, g, method = "spearman")) else NA_real_
  prs <- if (ok) suppressWarnings(cor(s, g, method = "pearson"))  else NA_real_
  data.frame(code = code_of[[d]], domain = unname(display_of[[d]]),
             sentence_precision = round(unname(spr[d]), 3),
             sentence_recall    = round(unname(src_[d]), 3),
             sentence_f1        = round(unname(sf1[d]), 3),
             sentence_support   = unname(ssup[d]),
             report_pearson = round(prs, 3), report_spearman = round(rho, 3),
             silver_mean_prop = round(mean(s), 3), gold_mean_prop = round(mean(g), 3),
             tier_by_spearman = tier(rho), stringsAsFactors = FALSE)
}))
s3 <- s3[order(match(s3$code, codes)), ]
s3$group <- substr(s3$code, 1, 1)          # A = ASD-core, B = general child-psychiatric, C = format

# per-group median Spearman rows appended below the per-domain rows
grp_med <- do.call(rbind, lapply(c("A", "B", "C"), function(g) {
  data.frame(code = g, domain = paste0(group_of[[g]], " (median)"),
             sentence_precision = NA_real_, sentence_recall = NA_real_,
             sentence_f1 = round(median(s3$sentence_f1[s3$group == g], na.rm = TRUE), 3),
             sentence_support = NA_integer_,
             report_pearson = NA_real_,
             report_spearman = round(median(s3$report_spearman[s3$group == g], na.rm = TRUE), 3),
             silver_mean_prop = NA_real_, gold_mean_prop = NA_real_,
             tier_by_spearman = NA_character_, group = g, stringsAsFactors = FALSE)
}))
out <- rbind(s3, grp_med)
out$group <- unname(group_of[out$group])   # spell the group out, as in Table 1 and the figure legends
write_table(out, "TableS3")
