#!/usr/bin/env Rscript
# Fig 6 — prototype + two-layer fingerprint
#   (a) ASD prototype domain composition
#   (b) within-vs-between variance decomposition, 2-way (core/periphery) + null only
#   (c) individual fingerprint
rm(list = ls())
source("_common.R")

dom <- domain_columns()
df  <- read.delim(TSV_PROP, sep = "\t", check.names = FALSE)
asd <- df[df$asd_label == 1, ]
Pmat <- as.matrix(asd[, dom]); storage.mode(Pmat) <- "double"

# (a) prototype composition
proto <- colMeans(Pmat)
oa <- order(proto)
da <- data.frame(name = factor(dlabel(dom[oa]), levels = dlabel(dom[oa])),
                 val = proto[oa] * 100,
                 grp = dgroup(dom[oa]))
pa <- ggplot(da, aes(val, name, fill = grp)) +
  geom_col(colour = "white", width = 0.78) +
  geom_text(aes(label = sprintf("%.1f", val)), hjust = -0.15, size = 2.82) +
  scale_fill_manual(values = GROUP_FILL,
                    labels = function(x) vapply(x, function(s)
                      paste(strwrap(s, width = 15), collapse = "\n"), character(1)),
                    name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.22))) +
  labs(x = "Domain composition of report's\nhigh-attention sentences (%)", y = NULL, tag = "a") +
  theme_all() + theme(legend.position = c(0.99, 0.23), legend.justification = c(1, 0.5),
                     legend.text = element_text(size = 8), legend.key.size = unit(0.30, "cm"),
                     legend.spacing.y = unit(0.05, "cm"), axis.text.y = element_text(size = 8))

# (b) variance decomposition (2-way + null)
v  <- fromJSON(VAR_JSON)
k2 <- v$residual_GMM_k2_core_vs_periphery
between <- k2$full_eta2_between            # between-mode variance share
nullv   <- k2$full_null_eta2_mean          # label-permutation null
db <- data.frame(grp = "2-way\n(core / psychiatry)",
                 part = factor(c("between", "within"), levels = c("between", "within")),
                 val  = c(between, 1 - between))
pb <- ggplot(db, aes(grp, val, fill = part)) +
  geom_col(width = 0.70) +
  geom_text(data = data.frame(grp = "2-way\n(core / psychiatry)", lab = sprintf("%.1f%%", between * 100)),
            aes(grp, 1, label = lab), inherit.aes = FALSE, colour = "black",
            fontface = "bold", size = 2.82, vjust = 2.4) +
  geom_text(data = data.frame(grp = "2-way\n(core / psychiatry)", lab = sprintf("%.1f%%", (1 - between) * 100)),
            aes(grp, 0.5, label = lab), inherit.aes = FALSE, colour = "black",
            fontface = "bold", size = 2.82) +
  geom_hline(yintercept = nullv, linetype = "dotted", colour = "#888", linewidth = 0.4) +
  annotate("text", x = 1, y = nullv + 0.03, label = sprintf("null %.1f%%", nullv * 100),
           size = 2.82, colour = "black", hjust = 0.5) +
  scale_fill_manual(values = c(between = PAL$red, within = PAL$panel),
                    labels = c(between = "between-group (coarse position)", within = "within-group (individual deviation)"),
                    name = NULL, breaks = c("between", "within")) +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.05))) +
  labs(x = NULL, y = "ASD variance share", tag = "b") +
  theme_all() + theme(legend.position = "right", legend.direction = "vertical",
                     legend.text = element_text(size = 8), legend.key.size = unit(0.32, "cm"),
                     axis.text.x = element_text(size = 8),
                     axis.title.y = element_text(size = 8, margin = margin(r = 1)),
                     plot.margin = margin(t = 8, r = 6, b = 2, l = 2))

# (c) Blank cell for the individual fingerprint exemplar.
pc <- empty_panel(tag = "c", note = "schematic") + theme(aspect.ratio = 540 / 680)

# assemble
row1 <- pa + pb + plot_layout(widths = c(1.7, 1))
fig  <- row1 / pc + plot_layout(heights = c(1, 1.5))

save_fig(fig, "Figure6", width = 7.0866, height = 7.6)
 