#!/usr/bin/env Rscript
# Supplementary FigureS3 — attention selection is a faithful, interpretable top-k
# compression of the report phenotype, not a performance-optimal filter.
# (a) cosine to expert-gold profile across k for top-k vs random-k; full-report ceiling.
# (b) cosine to the report's OWN full-sentence silver vector at k=10 (top-k vs random-k).
rm(list = ls())
source("../_common.R")
FS_DIR <- Sys.getenv("REPORT_LLM_FIGURE_SOURCE_DIR", FS_DIR)
OUT_DIR <- Sys.getenv("REPORT_LLM_SUPPLEMENTARY_FIGURE_OUT_DIR", OUT_DIR)

FS  <- 8
TAG <- 8
GEOM <- FS / .pt

DATA <- file.path(FS_DIR, "attention_faithfulness_test1_results.json")

C_TOP  <- PAL$blue
C_RAND <- PAL$grey
C_FULL <- PAL$orange

d  <- jsonlite::fromJSON(DATA, simplifyVector = FALSE)
ks <- as.integer(unlist(d$config$k_sweep))
kp <- as.integer(d$config$k_primary)

g <- d$gold_reference
top_g  <- vapply(ks, function(k) g[[as.character(k)]]$cosine$topK_median,  numeric(1))
rnd_g  <- vapply(ks, function(k) g[[as.character(k)]]$cosine$randK_median, numeric(1))
full_g <- g[[as.character(ks[1])]]$cosine$full_median            # constant ceiling
p_kp   <- g[[as.character(kp)]]$cosine$topK_vs_randK$wilcoxon_p

f     <- d$full_reference[[as.character(kp)]]$cosine
top_f <- f$topK_median
rnd_f <- f$randK_median

cat(sprintf("  gold k%d: top %.3f rand %.3f full %.3f (p=%.2f) | full-ref k%d: top %.3f rand %.3f\n",
            kp, top_g[ks == kp], rnd_g[ks == kp], full_g, p_kp, kp, top_f, rnd_f))

type_scale <- theme(
  text       = element_text(family = FONT, size = FS, colour = "black"),
  axis.text  = element_text(family = FONT, size = FS, colour = "black"),
  axis.title = element_text(family = FONT, size = FS, colour = "black"),
  plot.tag   = element_text(family = FONT, size = TAG, face = "bold")
)

# (a) cosine to expert-gold across k
line_df <- rbind(
  data.frame(k = ks, cos = top_g, grp = "top-k (attention)"),
  data.frame(k = ks, cos = rnd_g, grp = "random-k")
)
line_df$grp <- factor(line_df$grp, levels = c("top-k (attention)", "random-k"))
ylo <- min(top_g) - 0.02; yhi <- full_g + 0.055

pa <- ggplot(line_df, aes(k, cos, colour = grp, shape = grp, linetype = grp)) +
  geom_hline(yintercept = full_g, colour = C_FULL, linetype = "dashed", linewidth = 0.5) +
  geom_vline(xintercept = kp, colour = "#cccccc", linetype = "dotted", linewidth = 0.36) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.9) +
  annotate("text", x = ks[length(ks)], y = full_g + 0.008, label = "full report (all sentences)",
           colour = C_FULL, hjust = 1, vjust = 0, size = 8 / .pt, family = FONT) +
  annotate("text", x = kp + 0.4, y = min(top_g) - 0.004,
           label = sprintf("k=%d used\n(p=%.2f, n.s.)", kp, p_kp),
           colour = "#555555", hjust = 0, vjust = 1, size = 8 / .pt, family = FONT,
           lineheight = 0.95) +
  scale_colour_manual(values = c("top-k (attention)" = C_TOP, "random-k" = C_RAND), name = NULL) +
  scale_shape_manual(values  = c("top-k (attention)" = 16, "random-k" = 15), name = NULL) +
  scale_linetype_manual(values = c("top-k (attention)" = "solid", "random-k" = "dashed"), name = NULL) +
  scale_x_continuous(breaks = ks) +
  coord_cartesian(ylim = c(ylo, yhi)) +
  labs(x = "sentences selected (k)", y = "cosine to expert-gold profile", tag = "a") +
  theme_all(FS) + type_scale +
  theme(legend.position = c(0.98, 0.02), legend.justification = c(1, 0),
        legend.text = element_text(family = FONT, size = FS),
        legend.key = element_blank(), legend.background = element_blank(),
        legend.key.height = unit(0.42, "cm"),
        plot.margin = margin(8, 8, 6, 6))

# (b) cosine to full-report vector at k=primary
bar_df <- data.frame(grp = factor(c("top-k", "random-k"), levels = c("top-k", "random-k")),
                     cos = c(top_f, rnd_f))
pb <- ggplot(bar_df, aes(grp, cos, fill = grp)) +
  geom_col(width = 0.6, colour = "white") +
  geom_text(aes(label = sprintf("%.2f", cos)), vjust = -0.5, fontface = "bold",
            size = 8 / .pt, family = FONT) +
  scale_fill_manual(values = c("top-k" = C_TOP, "random-k" = C_RAND), guide = "none") +
  scale_y_continuous(limits = c(0, 1.05), expand = expansion(c(0, 0))) +
  labs(x = NULL, y = "cosine to full-report vector (k=10)", tag = "b") +
  theme_all(FS) + type_scale +
  theme(plot.margin = margin(8, 6, 6, 8))

fig <- pa + pb + patchwork::plot_layout(widths = c(1.55, 1))

save_fig(fig, "FigureS3", width = 7.0866, height = 2.98)
