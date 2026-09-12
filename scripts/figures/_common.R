#!/usr/bin/env Rscript
# _common.R — shared paths, theme, and helpers
rm(list = ls())
suppressWarnings(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"))

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(jsonlite)
})

.script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else getwd()
}
SCRIPT_DIR <- .script_dir()
REPO_ROOT <- normalizePath(file.path(SCRIPT_DIR, "..", ".."), mustWork = FALSE)
OUT_DIR <- SCRIPT_DIR

FS_DIR <- Sys.getenv("REPORT_LLM_FIGURE_SOURCE_DIR", file.path(REPO_ROOT, "figure_source"))
.fig_out <- file.path(OUT_DIR, "..", "Figures")
if (dir.exists(.fig_out)) OUT_DIR <- normalizePath(.fig_out)
TSV_PROP   <- file.path(FS_DIR, "domain_vectors_proportion_latest.tsv")
META_JSON  <- file.path(FS_DIR, "domain_vectors_meta_latest.json")
VAR_JSON   <- file.path(FS_DIR, "step9_variance_results.json")   # Fig 6 variance
PROFILE_TSV<- file.path(FS_DIR, "controlled_individual_profiles.tsv")

# palette
PAL <- list(
  navy = "#264653", teal = "#2A9D8F", coral = "#E76F51",
  blue = "#35618F", red = "#C15B4E", grey = "#8896A8",
  orange = "#EA7E4B", gold = "#E0B33A", green = "#3B9163",
  brown = "#A9744F", dblue = "#1F3F66",
  panel = "#DAE0E8"
)

# canonical domain display labels
DOMAIN_LABEL <- c(
  social_emotional_reciprocity = "Social-emotional reciprocity",
  nonverbal_communication      = "Nonverbal communication",
  relationship_play            = "Social relationships",
  stereotyped_behavior         = "Stereotyped behavior",
  insistence_on_sameness       = "Insistence on sameness",
  restricted_interests         = "Restricted interests",
  sensory_processing           = "Sensory reactivity",
  externalizing                = "Externalizing behaviors",
  internalizing                = "Internalizing behaviors",
  language_skills              = "Language skills",
  physiological_function       = "Physiological function",
  adaptive_behavior            = "Adaptive behavior",
  intelligence_learning        = "Intellectual functioning/learning skills",
  executive_function           = "Executive function",
  motor_skills                 = "Motor skills",
  family_environment           = "Family environment",
  test_scores                  = "Test scores",
  other_general                = "Other/general",
  recommendations              = "Recommendations")

# domain -> group code (CO/AS/RE) and group display labels
DOMAIN_GROUP <- setNames(rep(c("CO", "AS", "RE"), c(7, 9, 3)), names(DOMAIN_LABEL))
GROUP_LABEL  <- c(CO = "Core ASD domains",
                  AS = "Associated and co-occurring features",
                  RE = "Report elements and other")

# categorical group palette
# Colourblind-aware qualitative: CO coral / AS periwinkle / RE grey
GROUP_PAL  <- c(CO = "#E28E6D", AS = "#8DA0CB", RE = "#9E9E9E")
GROUP_FILL <- setNames(unname(GROUP_PAL), unname(GROUP_LABEL))  # keyed by display label

# helpers: internal id vector -> display label / group factor
dlabel <- function(ids) unname(DOMAIN_LABEL[ids])
dgroup <- function(ids) factor(unname(DOMAIN_GROUP[ids]), levels = c("CO", "AS", "RE"),
                               labels = GROUP_LABEL)

domain_columns <- function() jsonlite::fromJSON(META_JSON)$domain_columns

FONT <- "Arial"
suppressWarnings({
  update_geom_defaults("text",  list(family = FONT))   # in-panel annotations -> Arial
  update_geom_defaults("label", list(family = FONT))
})

# base theme
theme_all <- function(base = 8) {
  theme_grey(base_size = base, base_family = FONT) +
    theme(
      text = element_text(size = 8, family = FONT),
      legend.background = element_blank(),
      legend.title = element_text(size = 8),
      legend.text = element_text(size = 8),
      legend.key = element_blank(),
      legend.key.size = unit(0.38, "cm"),
      legend.margin = margin(t = .1, r = .1, b = .1, l = .1, unit = "cm"),
      legend.position = "none",
      
      axis.text.x = element_text(size = 8, color = "black"),
      axis.text.y = element_text(size = 8, color = "black"),
      axis.title.y = element_text(size = 8),
      axis.title.x = element_text(size = 8),
      axis.ticks = element_blank(),
      axis.line = element_line(colour = "black", linewidth = 1 / .pt),
      
      plot.title = element_text(size = 10, vjust = 1.5),
      plot.tag = element_text(size = 8, face = "bold", family = FONT),
      
      strip.text = element_text(face = "bold", size = 8),
      panel.background = element_blank(),
      panel.border = element_blank(),
      panel.grid = element_blank()
    )
}

# empty placeholder panel
empty_panel <- function(tag = NULL, note = "data not available yet") {
  ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = note, family = FONT,
             size = 3, colour = PAL$grey, fontface = "italic") +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    labs(tag = tag) +
    theme_void(base_size = 8, base_family = FONT) +
    theme(
      plot.tag = element_text(size = 8, face = "bold", family = FONT),
      panel.border = element_rect(colour = PAL$grey, fill = NA,
                                  linewidth = 0.5, linetype = "dashed"),
      plot.margin = margin(6, 6, 6, 6)
    )
}

# ILR transform (Helmert basis)
helmert_basis <- function(D) {
  V <- matrix(0, D, D - 1)
  for (i in seq_len(D - 1)) {
    s <- sqrt(i / (i + 1))          # sqrt((j+1)/(j+2)) with j=i-1
    V[seq_len(i), i] <- s / i
    V[i + 1, i] <- -s
  }
  V
}

ilr_transform <- function(X, pseudocount = 1e-6) {
  Xa <- X + pseudocount
  Xa <- Xa / rowSums(Xa)
  L  <- log(Xa)
  (L - rowMeans(L)) %*% helmert_basis(ncol(X))   # centering is a no-op for ILR; kept for parity
}

# report type indicator P from report_id last segment first char (A=0, P=1)
ptype_from_id <- function(rid) {
  last <- vapply(strsplit(rid, "-"), function(p) p[length(p)], character(1))
  as.integer(substr(last, 1, 1) == "P")
}

# residualize each ILR column on [1, P-indicator]; returns residual matrix
residualize_on_ptype <- function(ILR, rid) {
  design <- cbind(1, ptype_from_id(rid))
  RES <- matrix(0, nrow(ILR), ncol(ILR))
  XtX_inv_Xt <- solve(crossprod(design), t(design))
  for (j in seq_len(ncol(ILR))) {
    b <- XtX_inv_Xt %*% ILR[, j]
    RES[, j] <- ILR[, j] - design %*% b
  }
  RES
}

eff_dim <- function(evr) {
  p <- evr / sum(evr)
  exp(-sum(p * log(p)))
}

save_fig <- function(plot, file, width, height, dpi = 300) {
  base <- tools::file_path_sans_ext(file)
  png_out <- file.path(OUT_DIR, paste0(base, ".png"))
  pdf_out <- file.path(OUT_DIR, paste0(base, ".pdf"))
  # ragg::agg_png renders Unicode glyphs (rho, Delta, >=) with system fonts,
  # which the default png device drops (they show up as ".." / "...").
  ggsave(png_out, plot, width = width, height = height, dpi = dpi, bg = "white",
         device = ragg::agg_png)
  # cairo_pdf: vector PDF, embeds Arial, handles the same Unicode glyphs.
  ggsave(pdf_out, plot, width = width, height = height, bg = "white",
         device = cairo_pdf, family = FONT)
  cat("  ->", png_out, "\n  ->", pdf_out, "\n")
}
