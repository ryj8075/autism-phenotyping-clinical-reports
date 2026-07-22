#!/usr/bin/env Rscript
# Supplementary FigureS2 — sentence-attention heatmaps (A-type vs P-type exemplars).
rm(list = ls())
suppressPackageStartupMessages({ library(reshape2) })
source("../_common.R")
FS_DIR <- Sys.getenv("REPORT_LLM_FIGURE_SOURCE_DIR", FS_DIR)
OUT_DIR <- Sys.getenv("REPORT_LLM_SUPPLEMENTARY_FIGURE_OUT_DIR", OUT_DIR)

# unified type scale
FS   <- 8                        # body text size (pt)
TAG  <- 8                        # panel tag (a/b)
GEOM <- FS / .pt                 # equivalent in-panel annotate size (mm)
type_scale <- theme(
  text         = element_text(family = FONT, size = FS, colour = "black"),
  axis.text    = element_text(family = FONT, size = FS, colour = "black"),
  axis.title   = element_text(family = FONT, size = FS, colour = "black"),
  legend.title = element_text(family = FONT, size = FS),
  legend.text  = element_text(family = FONT, size = FS),
  plot.tag     = element_text(family = FONT, size = TAG, face = "bold")
)

N_STC <- 153L   # report_max_length (config.json)

# minimal pure-R .npy reader
.npy_open <- function(path) {
  con <- file(path, "rb")
  stopifnot(rawToChar(readBin(con, "raw", 6)[2:6]) == "NUMPY")
  ver <- as.integer(readBin(con, "raw", 2)[1])
  hlen <- if (ver == 1) readBin(con, "integer", 1, size = 2, signed = FALSE, endian = "little")
          else          readBin(con, "integer", 1, size = 4, endian = "little")
  hdr <- rawToChar(readBin(con, "raw", hlen))
  shp <- gsub("[^0-9,]", "", regmatches(hdr, regexpr("'shape':\\s*\\(([^)]*)\\)", hdr)))
  nums <- as.integer(strsplit(shp, ",")[[1]])
  list(con = con, shape = nums[!is.na(nums)],
       offset = 6 + 2 + (if (ver == 1) 2 else 4) + hlen)
}
npy_slice_f8 <- function(path, i, n) {   # [n,n] float64 C-order slice i (0-based) of (N,n,n)
  h <- .npy_open(path); on.exit(close(h$con))
  seek(h$con, h$offset + as.double(i) * n * n * 8, origin = "start")
  matrix(readBin(h$con, "double", n * n, size = 8, endian = "little"), n, n, byrow = TRUE)
}
npy_vec_f8 <- function(path) {
  h <- .npy_open(path); on.exit(close(h$con))
  readBin(h$con, "double", prod(h$shape), size = 8, endian = "little")
}
npy_vec_i8small <- function(path) {      # int64 (small): low 32-bit word per entry
  h <- .npy_open(path); on.exit(close(h$con))
  w <- readBin(h$con, "integer", prod(h$shape) * 2L, size = 4, endian = "little")
  w[seq(1, length(w), by = 2)]
}
npy_vec_strU <- function(path, width) {  # numpy '<U{width}' UTF-32LE fixed-width strings
  h <- .npy_open(path); on.exit(close(h$con))
  N <- prod(h$shape); b <- readBin(h$con, "raw", N * width * 4)
  vapply(seq_len(N), function(k) {
    cps <- as.integer(b[(k - 1) * width * 4 + seq(1, width * 4, by = 4)])
    rawToChar(as.raw(cps[cps != 0]))
  }, character(1))
}

BLUES <- c("#f8fafc", "#dae3ed", "#bdccdd", "#9ab1ca", "#6e8eb1",
           "#436c97", "#2f5884", "#274b75", "#1f3f66")   # toned, anchored on house navy #35618F

heatmap_panel <- function(M, nv, lab, tag = NULL) {
  sub <- M[seq_len(nv), seq_len(nv), drop = FALSE]
  keycol <- which.max(colSums(M))             # most-attended key sentence (all query rows)
  df <- reshape2::melt(sub, varnames = c("query", "key"), value.name = "w")
  ggplot(df, aes(key, query, fill = w)) +
    geom_raster() +
    annotate("rect", xmin = keycol - 0.5, xmax = keycol + 0.5,
             ymin = 0.5, ymax = nv + 0.5, fill = NA, colour = "black", linewidth = 0.45) +
    annotate("text", x = nv / 2, y = 0.5, label = lab, vjust = -0.6,
             fontface = "bold", size = GEOM, family = FONT) +
    scale_fill_gradientn(colours = BLUES, limits = c(0, 1),
                         breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1"),
                         name = "Attention weight") +
    scale_x_continuous(expand = expansion(0)) +
    scale_y_reverse(expand = expansion(0)) +
    coord_fixed(clip = "off") +
    labs(x = "Key sentence index", y = "Query sentence index", tag = tag) +
    theme_all(FS) +
    type_scale +
    theme(
      legend.position = "right",
      legend.title = element_text(family = FONT, size = FS, hjust = 0,
                                  margin = margin(b = 7)),   # lift title off the bar
      legend.key.width  = unit(0.22, "cm"),
      legend.key.height = unit(0.7, "cm"),
      plot.margin = margin(13, 4, 4, 4)
    )
}

build_heatmaps <- function() {
  exemplars <- file.path(FS_DIR, "attention_exemplars_np.npy")   # (2,153,153): [1]=A, [2]=P
  if (!file.exists(exemplars)) return(list(empty_panel("a", "attention data missing"), empty_panel("b")))
  probs <- npy_vec_f8(file.path(FS_DIR, "probs_np.npy"))
  nval  <- npy_vec_i8small(file.path(FS_DIR, "n_valid_sentences_np.npy"))
  rid   <- npy_vec_strU(file.path(FS_DIR, "report_id_array.npy"), 16)
  typ <- substr(vapply(strsplit(rid, "-"), function(p) p[length(p)], ""), 1, 1)
  pick <- function(t) { idx <- which(typ == t); idx[which.max(probs[idx])] }
  aBest <- pick("A"); pBest <- pick("P")
  # slim file holds exactly the two selected matrices, in A,P order (same selection rule)
  Ma <- npy_slice_f8(exemplars, 0L, N_STC)
  Mp <- npy_slice_f8(exemplars, 1L, N_STC)
  cat(sprintf("  exemplar A: %s (P=%.3f, n=%d)\n", rid[aBest], probs[aBest], nval[aBest]))
  cat(sprintf("  exemplar P: %s (P=%.3f, n=%d)\n", rid[pBest], probs[pBest], nval[pBest]))
  list(
    heatmap_panel(Ma, nval[aBest], sprintf("A-type  (%s, P = %.3f)", rid[aBest], probs[aBest]), tag = "a"),
    heatmap_panel(Mp, nval[pBest], sprintf("P-type  (%s, P = %.3f)", rid[pBest], probs[pBest]), tag = "b")
  )
}

hp  <- build_heatmaps()
fig <- hp[[1]] + hp[[2]]

save_fig(fig, "FigureS2", width = 7.0866, height = 3.46)
