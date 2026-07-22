#!/usr/bin/env Rscript
# Fig 2 — diagnostic AUROC by report type
rm(list = ls())
source("_common.R")

# unified type scale
FS   <- 8                        # body text size (pt) — 8pt at print size
GEOM <- FS / .pt                 # equivalent in-panel geom_text size (mm)
type_scale <- theme(
  text         = element_text(family = FONT, size = FS, colour = "black"),
  axis.text    = element_text(family = FONT, size = FS, colour = "black"),
  axis.title   = element_text(family = FONT, size = FS, colour = "black"),
  axis.text.x  = element_text(family = FONT, size = FS, colour = "black"),
  axis.text.y  = element_text(family = FONT, size = FS, colour = "black"),
  axis.title.x = element_text(family = FONT, size = FS, colour = "black"),
  axis.title.y = element_text(family = FONT, size = FS, colour = "black")
)

# AUROC bars
# Values = 5-fold CV mean +/- sd from each model's results
read_cv_metric <- function(path, metric = "AUROC") {
  L <- readLines(path, warn = FALSE)
  pat <- sprintf("%s:\\s*([0-9.]+)\\s*\\(\\+/-\\s*([0-9.]+)\\)", metric)
  for (l in L) {
    m <- regmatches(l, regexec(pat, l))[[1]]
    if (length(m) == 3) return(c(mean = as.numeric(m[2]), sd = as.numeric(m[3])))
  }
  stop(sprintf("Could not find a '%s' summary line in %s.", metric, basename(path)))
}

prefix <- c(All = "489samples", A = "asd233samples", P = "psy256samples")
suffix <- "_epoch40_153stc_128tkn_epoch40_patience10_no_headings.txt"
vals <- t(vapply(prefix, function(p) read_cv_metric(file.path(FS_DIR, paste0(p, suffix)), "AUROC"), numeric(2)))

.lab <- c("All\n(n=489)",
          "Autism diagnosis report\n(n=233)",
          "Psychological assessment report\n(n=256)")
da <- data.frame(
  group = factor(.lab, levels = .lab),
  auroc = vals[, "mean"],
  sd    = vals[, "sd"],
  col   = c("#3E5C7E", "#6BA6A0", "#C6A85C")   # All navy / A-type teal / P-type gold (report-type palette, distinct from domain-group colours)
)
# value label sits just above each error-bar top
da$labely <- da$auroc + da$sd + 0.03

fig <- ggplot(da, aes(group, auroc)) +
  geom_col(aes(fill = col), width = 0.6, colour = "white") +
  geom_errorbar(aes(ymin = auroc - sd, ymax = auroc + sd), width = 0.18, linewidth = 0.5) +
  geom_hline(yintercept = 0.5, linetype = "dotted", colour = "#999", linewidth = 0.4) +
  geom_text(aes(y = labely, label = sprintf("%.3f", auroc)),
            fontface = "bold", size = GEOM, family = FONT) +
  scale_fill_identity() +
  scale_y_continuous(breaks = seq(0, 1.0, 0.25), expand = expansion(mult = c(0, 0.02))) +
  coord_cartesian(ylim = c(0, 1.08)) +
  labs(x = NULL, y = "AUROC (5-fold CV)") +
  theme_all(FS) +
  type_scale

save_fig(fig, "Figure2", width = 7.0866, height = 3.6)
