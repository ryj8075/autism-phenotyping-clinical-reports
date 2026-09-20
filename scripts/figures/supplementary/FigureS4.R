#!/usr/bin/env Rscript
# Supplementary FigureS4 — report type as a confound and its removal.
# (a) share of each domain's CLR variance explained by report type across the 346 ASD
#    reports, with Benjamini-Hochberg FDR.
# (b) first principal component of the ILR representation before and after
#    type-adjustment, split by documentation stream. The separation before adjustment
#    is the confound; after adjustment PC1 is orthogonal to report type
#    by construction.
rm(list = ls())
source("../_common.R")
FS_DIR <- Sys.getenv("REPORT_LLM_FIGURE_SOURCE_DIR", FS_DIR)
OUT_DIR <- Sys.getenv("REPORT_LLM_SUPPLEMENTARY_FIGURE_OUT_DIR", OUT_DIR)

FS <- 8
d <- jsonlite::fromJSON(file.path(FS_DIR, "report_type_confound.json"))

# (a) per-domain variance explained by report type
a_df <- data.frame(
  domain = d$panel_a$domain,
  r2     = as.numeric(d$panel_a$r2_report_type) * 100,
  sig    = as.logical(d$panel_a$significant_q10)
)
a_df$label <- ifelse(is.na(DOMAIN_LABEL[a_df$domain]), a_df$domain, DOMAIN_LABEL[a_df$domain])
a_df$group <- dgroup(a_df$domain)
a_df <- a_df[order(a_df$r2), ]
a_df$label <- factor(a_df$label, levels = a_df$label)

pa <- ggplot(a_df, aes(label, r2, fill = group)) +
  geom_col(width = 0.74, colour = "white", linewidth = 0.18) +
  geom_point(data = subset(a_df, sig), aes(y = r2 + 1.4), shape = 8,
             size = 0.9, stroke = 0.35, colour = "#333333", inherit.aes = TRUE,
             show.legend = FALSE) +
  scale_fill_manual(values = GROUP_FILL, name = NULL) +
  scale_y_continuous(limits = c(0, max(a_df$r2) * 1.16), expand = expansion(c(0, 0))) +
  coord_flip() +
  labs(x = NULL, y = "variance explained by report type (%)") +
  theme_all(FS) +
  theme(legend.position = c(0.985, 0.03),
        legend.justification = c(1, 0),
        legend.direction = "vertical",
        legend.key.size = unit(6, "pt"),
        legend.key.spacing.y = unit(0.5, "pt"),
        legend.text = element_text(size = FS, family = FONT),
        legend.background = element_rect(fill = "white", colour = NA),
        legend.margin = margin(1, 2, 1, 1),
        axis.text.y = element_text(size = FS, family = FONT, colour = "black"),
        plot.margin = margin(6, 8, 6, 4))

# (b) PC1 before and after adjustment
b_df <- rbind(
  data.frame(stage = "unadjusted", type = d$panel_b$report_type,
             pc1 = as.numeric(d$panel_b$pc1_raw)),
  data.frame(stage = "type-adjusted", type = d$panel_b$report_type,
             pc1 = as.numeric(d$panel_b$pc1_adjusted))
)
b_df$stage <- factor(b_df$stage, levels = c("unadjusted", "type-adjusted"))
b_df$type  <- factor(b_df$type, levels = c("A", "P"),
                     labels = c("A-type", "P-type"))

r_raw <- d$panel_b$corr_pc1_type_raw
r_adj <- d$panel_b$corr_pc1_type_adjusted
ann <- data.frame(
  stage = factor(c("unadjusted", "type-adjusted"),
                 levels = c("unadjusted", "type-adjusted")),
  lab = c(sprintf("r = %.2f", abs(r_raw)), sprintf("r = %.2f", abs(r_adj))),
  y = max(b_df$pc1) * 1.06
)

pb <- ggplot(b_df, aes(type, pc1, fill = type)) +
  geom_boxplot(width = 0.56, outlier.size = 0.35, outlier.colour = "#777777",
               linewidth = 0.3, colour = "#333333", alpha = 0.9) +
  geom_text(data = ann, aes(x = 1.5, y = y, label = lab), inherit.aes = FALSE,
            family = FONT, size = FS / .pt, colour = "#333333") +
  facet_wrap(~ stage, nrow = 1) +
  # report-type colours identical to Figure 2 (A-type teal, P-type gold)
  scale_fill_manual(values = c(`A-type` = "#6BA6A0", `P-type` = "#C6A85C"), guide = "none") +
  scale_y_continuous(expand = expansion(c(0.05, 0.14))) +
  labs(x = NULL, y = "PC1 score") +
  theme_all(FS) +
  theme(strip.background = element_blank(),
        strip.text = element_text(size = FS, family = FONT, colour = "black"),
        plot.margin = margin(6, 4, 6, 8))

fig <- (pa | pb) + patchwork::plot_layout(widths = c(1.22, 1)) +
  patchwork::plot_annotation(tag_levels = "a")

save_fig(fig, "FigureS4", width = 7.0866, height = 4.1)
