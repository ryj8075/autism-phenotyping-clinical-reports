#!/usr/bin/env Rscript
# Supplementary FigureS6
# (a) chi-square QQ of the full Mahalanobis distance.
# (b) prototype + heavy-tail in ONE scatter (distance-distance plot).
rm(list = ls())
source("../_common.R")
FS_DIR <- Sys.getenv("REPORT_LLM_FIGURE_SOURCE_DIR", FS_DIR)
OUT_DIR <- Sys.getenv("REPORT_LLM_SUPPLEMENTARY_FIGURE_OUT_DIR", OUT_DIR)
TSV_PROP  <- file.path(FS_DIR, "domain_vectors_proportion_latest.tsv")
META_JSON <- file.path(FS_DIR, "domain_vectors_meta_latest.json")
suppressPackageStartupMessages(library(MASS))

K_LEAD <- 3                                   # leading subspace = PC1..PC3 (matches step4 split)

dom  <- domain_columns()
df   <- read.delim(TSV_PROP, sep = "\t", check.names = FALSE)
asd  <- df[df$asd_label == 1, ]
Pmat <- as.matrix(asd[, dom]); storage.mode(Pmat) <- "double"
rid  <- asd$report_id

# type-residual ILR -> PCA (identical to main distribution recompute)
ILR    <- ilr_transform(Pmat)
RES    <- residualize_on_ptype(ILR, rid)
pc     <- prcomp(RES, center = TRUE, scale. = FALSE)
scores <- pc$x
p      <- ncol(scores)
lead   <- scores[, 1:K_LEAD, drop = FALSE]
resid  <- scores[, (K_LEAD + 1):p, drop = FALSE]

# per-subspace Mahalanobis (PC scores are uncorrelated -> blocks separate)
d_lead <- sqrt(mahalanobis(lead,  colMeans(lead),  cov(lead)))
d_res  <- sqrt(mahalanobis(resid, colMeans(resid), cov(resid)))
d_full <- sqrt(mahalanobis(scores, colMeans(scores), cov(scores)))

# Gaussian reference: under normality each subspace distance^2 ~ chi-square(df)
ref_res  <- sqrt(qchisq(0.99, df = ncol(resid)))
ref_lead <- sqrt(qchisq(0.99, df = K_LEAD))

dd <- data.frame(d_lead = d_lead, d_res = d_res, d_full = d_full,
                 rid = rid, tail = d_res > ref_res)

# an idiosyncratic exemplar: ordinary leading distance, large residual (heavy-tail) distance
EX <- "REPORT-066-A01"
ex <- dd[dd$rid == EX, ][1, ]

cat(sprintf("n=%d  ILR dims=%d  (lead=%d, resid=%d)\n", nrow(dd), p, K_LEAD, ncol(resid)))
cat(sprintf("residual Gaussian 99%%: %.2f | reports above it: %d (%.0f%%)\n",
            ref_res, sum(dd$tail), 100 * mean(dd$tail)))
cat(sprintf("exemplar %s: d_lead=%.2f d_res=%.2f d_full=%.2f\n",
            EX, ex$d_lead, ex$d_res, ex$d_full))

# (a) chi-square QQ of full Mahalanobis distance
maha2 <- d_full^2
ndim  <- p
nn    <- length(maha2)
qq    <- data.frame(q = qchisq((seq_len(nn) - 0.5) / nn, df = ndim), d2 = sort(maha2))
ks    <- suppressWarnings(ks.test(maha2, "pchisq", df = ndim))
lim   <- max(qq$q, qq$d2)
pp_qq <- ggplot(qq, aes(q, d2)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = PAL$grey, linewidth = 0.4) +
  geom_point(size = 1.4, alpha = 0.85, stroke = 0, colour = PAL$navy) +
  annotate("text", x = 0, y = lim,
           label = sprintf("KS D = %.3f\np = %.3f", ks$statistic, ks$p.value),
           family = FONT, size = 8 / .pt, hjust = 0, vjust = 1, lineheight = 0.95, colour = "#333") +
  coord_equal(xlim = c(0, lim), ylim = c(0, lim)) +
  labs(x = sprintf("theoretical chi-square quantile (df = %d)", ndim),
       y = "observed squared Mahalanobis distance", tag = "a") +
  theme_all()

# (b) prototype + heavy-tail in ONE scatter (distance-distance plot)
pp_dd <- ggplot(dd, aes(d_lead, d_res)) +
  geom_hline(yintercept = ref_res, linetype = "dashed", colour = PAL$red, linewidth = 0.4) +
  geom_vline(xintercept = ref_lead, linetype = "dotted", colour = PAL$grey, linewidth = 0.35) +
  geom_point(aes(colour = d_full), size = 1.5, alpha = 0.85, stroke = 0) +
  geom_point(data = ex, shape = 21, size = 2.6, stroke = 0.6,
             colour = "black", fill = NA) +
  annotate("text", x = min(dd$d_lead) - 0.2, y = ex$d_res + 0.70, label = "idiosyncratic exemplar",
           family = FONT, size = 8 / .pt, colour = "black", hjust = 0) +
  annotate("text", x = ref_lead - 0.08, y = max(dd$d_res) + 1.2,
           label = "leading\nGaussian 99%", family = FONT, size = 8 / .pt,
           colour = PAL$grey, hjust = 1, vjust = 1, lineheight = 0.9) +
  annotate("text", x = max(dd$d_lead) - 1.15, y = ref_res + 0.28,
           label = "residual Gaussian 99%\n(heavy-tail threshold)", family = FONT,
           size = 8 / .pt, colour = PAL$red, hjust = 0, vjust = 0, lineheight = 0.9) +
  scale_colour_gradientn(colours = c(PAL$gold, PAL$coral, PAL$navy),
                         name = "overall distance\n(prototypicality, L1)") +
  labs(x = "leading-subspace distance  (PC1-3, shared near-Gaussian core)",
       y = "residual-subspace distance  (PC4+, idiosyncratic tail)", tag = "b") +
  coord_cartesian(xlim = c(0, max(dd$d_lead) * 1.20),
                  ylim = c(0, max(dd$d_res) * 1.10), clip = "on") +
  theme_all() +
  theme(aspect.ratio = 1,
        legend.position = "right", legend.direction = "vertical",
        legend.title = element_text(size = 8), legend.text = element_text(size = 8),
        legend.key.height = unit(0.34, "cm"), legend.key.width = unit(0.30, "cm"))

fig <- pp_qq + pp_dd + plot_layout(widths = c(1, 1))
save_fig(fig, "FigureS6", width = 7.0866, height = 3.2)
