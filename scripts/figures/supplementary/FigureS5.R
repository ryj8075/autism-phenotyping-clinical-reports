#!/usr/bin/env Rscript
# Supplementary FigureS5 — A-type-only mode stability (report-type confound sensitivity).
rm(list = ls())
source("../_common.R")
FS_DIR <- Sys.getenv("REPORT_LLM_FIGURE_SOURCE_DIR", FS_DIR)
OUT_DIR <- Sys.getenv("REPORT_LLM_SUPPLEMENTARY_FIGURE_OUT_DIR", OUT_DIR)

FS <- 8

JSON <- file.path(FS_DIR, "atype_only_mode_stability.json")

d   <- jsonlite::fromJSON(JSON, simplifyVector = FALSE)
per_k <- d$per_k
ks  <- sort(as.integer(names(per_k)))
ari <- vapply(ks, function(k) as.numeric(per_k[[as.character(k)]]$seed_ari), numeric(1))
n_a <- if (!is.null(d$n)) as.integer(d$n) else NA_integer_
cat(sprintf("  A-only (n=%s) seed ARI: k2=%.2f k4=%.2f\n", n_a, ari[ks == 2], ari[ks == 4]))

df <- data.frame(k = factor(ks), ari = ari, stable = ks == 2)
ymax <- 1.20   # headroom for the 2-line "2-way / stable" label above the k=2 bar

fig <- ggplot(df, aes(k, ari, fill = stable)) +
  geom_col(width = 0.78, colour = "white", alpha = 0.9) +
  geom_hline(yintercept = 0.8, linetype = "dotted", colour = "#444444", linewidth = 0.36) +
  annotate("text", x = length(ks) - 0.3, y = 0.82, label = "stable >= 0.8",
           hjust = 0, vjust = 0, size = 8 / .pt, family = FONT, colour = "#444444") +
  annotate("text", x = 1, y = ari[ks == 2] + 0.03, label = "2-way\nstable",
           hjust = 0.5, vjust = 0, fontface = "bold", colour = PAL$blue,
           size = 8 / .pt, family = FONT, lineheight = 0.95) +
  scale_fill_manual(values = c(`TRUE` = PAL$blue, `FALSE` = PAL$grey), guide = "none") +
  scale_y_continuous(limits = c(0, ymax), expand = expansion(c(0, 0))) +
  labs(x = expression("mixture components " * italic(k)), y = "cross-seed ARI") +
  theme_all(FS) +
  theme(
    text       = element_text(family = FONT, size = FS, colour = "black"),
    axis.text  = element_text(family = FONT, size = FS, colour = "black"),
    axis.title = element_text(family = FONT, size = FS, colour = "black"),
    plot.margin = margin(8, 8, 6, 6)
  )

save_fig(fig, "FigureS5", width = 7.0866, height = 3.5)
