#!/usr/bin/env Rscript
# Supplementary Figure S1 — selector classifier characterization:
# (a) 489 5-fold confusion matrix
# (b) report-embedding PCA (coloured by MD diagnosis)
rm(list = ls())
source("../_common.R")
FS_DIR <- Sys.getenv("REPORT_LLM_FIGURE_SOURCE_DIR", FS_DIR)
OUT_DIR <- Sys.getenv("REPORT_LLM_SUPPLEMENTARY_FIGURE_OUT_DIR", OUT_DIR)
suppressPackageStartupMessages({ library(reshape2); library(patchwork) })

FS <- 8

# (a) confusion matrix (current 489 5-fold, summed over folds)
CM_TXT <- file.path(FS_DIR, "489samples_epoch40_153stc_128tkn_epoch40_patience10_no_headings.txt")
.cml <- readLines(CM_TXT, warn = FALSE)
.tp  <- grep("TP=", .cml)                 # per-fold lines + the summed-total line
if (!length(.tp)) stop("no TP=/FP=/TN=/FN= line found in: ", basename(CM_TXT))
.sum <- .cml[tail(.tp, 1)]
.get <- function(key) as.integer(sub(".*=", "", regmatches(.sum, regexpr(sprintf("%s=[0-9]+", key), .sum))))
TN <- .get("TN"); FP <- .get("FP"); FN <- .get("FN"); TP <- .get("TP")
cm <- matrix(c(TN, FP, FN, TP), nrow = 2, byrow = TRUE)
labs_cm <- c("Non-ASD", "ASD")
BLUES <- c("#f8fbfe", "#e0ebf5", "#cadbeb", "#a5c8da", "#76abcb", "#4f8fb9", "#3070a6", "#17518d", "#123261")
dca <- reshape2::melt(cm, varnames = c("true_i", "pred_j"), value.name = "n")
dca$true <- factor(labs_cm[dca$true_i], levels = rev(labs_cm))
dca$pred <- factor(labs_cm[dca$pred_j], levels = labs_cm)
dca$txtcol <- ifelse(dca$n > max(cm) / 2, "white", "#222222")
pa <- ggplot(dca, aes(pred, true, fill = n)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = n, colour = txtcol), fontface = "bold", size = 8 / .pt, family = FONT) +
  scale_colour_identity() + scale_fill_gradientn(colours = BLUES, guide = "none") +
  scale_x_discrete(expand = expansion(0)) + scale_y_discrete(expand = expansion(0)) +
  coord_fixed(clip = "off") +
  labs(x = "predicted", y = "true", tag = "a") +
  theme_all(FS) + theme(axis.line = element_blank(), plot.margin = margin(6, 6, 6, 6))

# (b) report-embedding PCA (coloured by MD diagnosis)
db <- read.csv(file.path(FS_DIR, "figS1_embedding_pca.csv"), stringsAsFactors = FALSE)
db$diagnosis <- factor(db$diagnosis, levels = c("Autism", "Non-autism"))
pc1 <- db$pc1_var_pct[1]; pc2 <- db$pc2_var_pct[1]
pb <- ggplot(db, aes(PC1, PC2, colour = diagnosis)) +
  geom_point(size = 1.5, alpha = 0.85, stroke = 0) +
  scale_colour_manual(values = c(Autism = "#9bd397", `Non-autism` = "#6c5688"), name = NULL) +
  labs(x = sprintf("PC1 (%.1f%%)", pc1), y = sprintf("PC2 (%.1f%%)", pc2), tag = "b") +
  guides(colour = guide_legend(override.aes = list(size = 2.4))) +
  theme_all(FS) +
  theme(legend.position = c(0.99, 0.99), legend.justification = c(1, 1),
        legend.background = element_rect(fill = alpha("white", 0.7), colour = NA),
        legend.key = element_blank(), legend.key.height = unit(0.34, "cm"),
        legend.text = element_text(size = FS), legend.title = element_blank(),
        legend.margin = margin(1, 2, 1, 2), legend.spacing.y = unit(0.02, "cm"),
        plot.margin = margin(6, 6, 6, 8))

fig <- pa + pb + plot_layout(widths = c(1, 1))
save_fig(fig, "FigureS1", width = 7.0866, height = 3.5433)  # 180 x 90 mm; two 90 x 90 mm panels
