#!/usr/bin/env Rscript
# Table S2 — per-fold AUROC and specificity (5-fold CV) for the three selector models
rm(list = ls())
source("_common.R")
RES <- TS_DIR
files <- c(All = "489samples")
suffix <- "_epoch40_153stc_128tkn_epoch40_patience10_no_headings.txt"

# per-fold lines look like:
#   Fold 1 (Best Epoch=7): AUROC=0.8794, AUPRC=0.9390, ..., Spec=0.5484, F1=0.8707
parse_metric <- function(tag, key) {
  p <- file.path(RES, paste0(files[[tag]], suffix)); stopifnot(file.exists(p))
  L <- grep("^Fold [0-9]+ ", readLines(p, warn = FALSE), value = TRUE)   # per-fold lines only
  as.numeric(sub(paste0(key, "="), "",
                 unlist(regmatches(L, gregexpr(paste0(key, "=[0-9.]+"), L)))))
}
AUR <- lapply(names(files), parse_metric, key = "AUROC"); names(AUR) <- names(files)
SPE <- lapply(names(files), parse_metric, key = "Spec");  names(SPE) <- names(files)

nf  <- max(vapply(AUR, length, integer(1)))
pad <- function(v) c(round(v, 3), rep(NA, nf - length(v)))
s3 <- data.frame(fold = seq_len(nf),
                 AUROC = pad(AUR$All), specificity = pad(SPE$All),
                 stringsAsFactors = FALSE)
write_table(s3, "TableS2")
