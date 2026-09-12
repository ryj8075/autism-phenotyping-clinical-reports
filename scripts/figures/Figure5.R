#!/usr/bin/env Rscript
# Fig 5 — no discrete subtypes; single heavy-tailed cloud with idiosyncrasy in the residual tail.
#   (a) GMM BIC vs multivariate-t: t beats every GMM k
#   (b) stability of the k-way split: cross-seed ARI + bootstrap ARI + permutation null
#   (c) Mahalanobis-to-prototype histogram vs Gaussian chi(df) reference
#   (d) heavy tail concentrates in residual PC4+ (Mardia z + 99th-pct excess, twin axis)
#   (e) Layer-2 cohort: domains carrying individual deviation (|z|>2)
rm(list = ls())
source("_common.R")

# per-step outputs of the type-residual phenotype-space analysis
STEP2_JSON <- file.path(FS_DIR, "step2_gmm_vs_heavytail_results.json")   # (a) GMM vs t
STEP6_JSON <- file.path(FS_DIR, "step6_seed_stability_results.json")     # (b) cross-seed ARI
STEP4_JSON <- file.path(FS_DIR, "step4_residual_tail_results.json")      # (d) tail decomposition
BOOT_JSON  <- file.path(FS_DIR, "bootstrap_permutation_results.json")    # (b) bootstrap + null

s2 <- fromJSON(STEP2_JSON)
s6 <- fromJSON(STEP6_JSON)
s4 <- fromJSON(STEP4_JSON)
bp <- fromJSON(BOOT_JSON)
ks <- sort(as.integer(names(s2$per_k)))
bics  <- vapply(ks, function(k) s2$per_k[[as.character(k)]]$gmm_bic, numeric(1))
t_bic <- s2$t_dist$bic
best_k <- ks[which.min(bics)]                 # GMM's own best (data-driven; = 2)
used_bic <- min(bics)
dbic <- s2$per_k[[as.character(best_k)]]$dbic_vs_t   # t wins => positive

# (a) BIC: multivariate-t beats every GMM k
pa <- ggplot(data.frame(k = ks, bic = bics), aes(k, bic)) +
  geom_hline(yintercept = t_bic, colour = PAL$grey, linetype = "dashed", linewidth = 0.5) +
  geom_line(colour = PAL$blue, linewidth = 0.6) +
  geom_point(colour = PAL$blue, size = 1.8) +
  geom_point(data = data.frame(k = best_k, bic = used_bic), colour = PAL$red, size = 3.2) +
  annotate("text", x = max(ks), y = t_bic, label = sprintf("mult. t (%.0f)", t_bic),
           colour = "#5b6672", size = 2.82, hjust = 1, vjust = 1.6) +
  annotate("text", x = best_k + 0.10, y = used_bic + 60,
           label = sprintf("best GMM (k=%d)\n%+0.0f BIC worse than t", best_k, dbic),
           colour = PAL$red, fontface = "bold", size = 2.82, hjust = 0, vjust = 0.4, lineheight = 0.9) +
  scale_x_continuous(breaks = ks) +
  coord_cartesian(ylim = c(min(c(bics, t_bic)) - 6, max(bics) + 6)) +
  labs(x = "mixture components k", y = "BIC (lower better)", tag = "a") +
  theme_all()

# (b) stability: cross-seed ARI + bootstrap ARI + permutation null
seed_m <- vapply(ks, function(k) s6$per_k[[as.character(k)]]$seed_ari_mean, numeric(1))
seed_s <- vapply(ks, function(k) s6$per_k[[as.character(k)]]$seed_ari_sd,   numeric(1))
boot_m <- vapply(as.character(ks), function(k) bp$per_k[[k]]$bootstrap$mean,    numeric(1))
boot_lo<- vapply(as.character(ks), function(k) bp$per_k[[k]]$bootstrap$ci95_lo, numeric(1))
boot_hi<- vapply(as.character(ks), function(k) bp$per_k[[k]]$bootstrap$ci95_hi, numeric(1))
null_lo<- vapply(as.character(ks), function(k) bp$per_k[[k]]$null$ci95_lo,      numeric(1))
null_hi<- vapply(as.character(ks), function(k) bp$per_k[[k]]$null$ci95_hi,      numeric(1))

off <- 0.16
db_seed <- data.frame(k = ks - off, m = seed_m, lo = pmax(0, seed_m - seed_s), hi = pmin(1, seed_m + seed_s), grp = "cross-seed ARI")
db_boot <- data.frame(k = ks + off, m = boot_m, lo = boot_lo, hi = boot_hi, grp = "bootstrap ARI")
db_pts  <- rbind(db_seed, db_boot)
db_pts$grp <- factor(db_pts$grp, levels = c("cross-seed ARI", "bootstrap ARI"))
db_null <- data.frame(k = ks, lo = null_lo, hi = null_hi)

pb <- ggplot() +
  # permutation null 95% band (essentially 0)
  geom_ribbon(data = db_null, aes(x = k, ymin = lo, ymax = hi), fill = PAL$grey, alpha = 0.5) +
  annotate("text", x = max(ks), y = 0.06, label = "permutation null (95%)",
           colour = "#5b6672", size = 2.82, hjust = 1, vjust = 0) +
  geom_hline(yintercept = 0.8, linetype = "dotted", colour = "#444", linewidth = 0.4) +
  annotate("text", x = max(ks), y = 0.85, label = "stable ≥ 0.8", size = 2.82, hjust = 1, colour = "#444") +
  geom_errorbar(data = db_pts, aes(x = k, ymin = lo, ymax = hi, colour = grp), width = 0.12, linewidth = 0.45) +
  geom_point(data = db_pts, aes(x = k, y = m, colour = grp), size = 1.7) +
  annotate("text", x = 1.55, y = 1.1, label = "coarse 2-way split only", size = 2.82, colour = PAL$blue,
           fontface = "bold", hjust = 0) +
  scale_colour_manual(values = c("cross-seed ARI" = PAL$blue, "bootstrap ARI" = PAL$red), name = NULL) +
  scale_x_continuous(breaks = ks) +
  coord_cartesian(ylim = c(-0.05, 1.15)) +
  labs(x = "mixture components k", y = "ARI vs reference", tag = "b") +
  theme_all() +
  theme(legend.position = c(0.99, 0.98), legend.justification = c(1, 1),
        legend.text = element_text(size = 8), legend.key.size = unit(0.30, "cm"),
        legend.background = element_blank())

# distribution recompute (type-residual ILR -> PCA -> Mahalanobis)
dom <- domain_columns()
df  <- read.delim(TSV_PROP, sep = "\t", check.names = FALSE)
asd <- df[df$asd_label == 1, ]
Pmat <- as.matrix(asd[, dom]); storage.mode(Pmat) <- "double"
rid  <- asd$report_id
ILR  <- ilr_transform(Pmat)
RES  <- residualize_on_ptype(ILR, rid)
pc   <- prcomp(RES, center = TRUE, scale. = FALSE)
scores <- pc$x
ndim <- ncol(scores)
maha2 <- mahalanobis(scores, colMeans(scores), cov(scores))
maha  <- sqrt(maha2)
chi99 <- sqrt(qchisq(0.99, df = ndim))

# (c) Mahalanobis-to-prototype histogram + chi(df) reference
dchi <- function(x, dfree) 2 * x * dchisq(x^2, df = dfree)   # density of chi(df): X=sqrt(chisq)
xg <- seq(0, max(maha) * 1.05, length.out = 400)
ref <- data.frame(x = xg, y = dchi(xg, ndim))
pc_hist <- ggplot(data.frame(m = maha), aes(m)) +
  geom_histogram(aes(y = after_stat(density)), bins = 26, fill = PAL$navy, alpha = 0.80,
                 colour = "white", linewidth = 0.36) +
  geom_line(data = ref, aes(x, y), colour = PAL$red, linewidth = 0.7) +
  geom_vline(xintercept = chi99, colour = PAL$grey, linetype = "dashed", linewidth = 0.4) +
  annotate("text", x = chi99, y = Inf, label = "Gaussian 99%", colour = "#5b6672",
           size = 2.82, hjust = -0.05, vjust = 1.4) +
  annotate("text", x = max(maha) * 0.08, y = max(ref$y) * 0.92,
           label = sprintf("Gaussian ref\nχ(df=%d)", ndim), colour = PAL$red,
           size = 2.82, hjust = 0, vjust = 1, lineheight = 0.9) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(x = "Mahalanobis distance to prototype", y = "density", tag = "c") +
  theme_all()

# ---- (d) tail by subspace (twin axis) ----------------------------------------
MARDIA_COL <- PAL$blue   # navy for Mardia z (matches panel a; teal/gold reserved for report type)
EXCESS_COL <- PAL$red
keys <- c("pc1_3", "pc4plus"); labk <- c("PC1-3", "PC4+")
z_vals <- vapply(keys, function(k) s4[[k]]$z, numeric(1))
excess <- vapply(keys, function(k) s4[[k]]$excess99, numeric(1))
zlo <- -8; zhi <- max(z_vals) * 1.18
e_to_z <- function(e) zlo + (e - 0) / (max(excess) * 1.1) * (zhi - zlo)
z_to_e <- function(z) (z - zlo) / (zhi - zlo) * (max(excess) * 1.1)
dd <- data.frame(grp = rep(labk, 2), metric = rep(c("z", "excess"), each = 2),
                 yval = c(z_vals, e_to_z(excess)),
                 xpos = c(1:2 - 0.18, 1:2 + 0.18),
                 col = rep(c(MARDIA_COL, EXCESS_COL), each = 2))
pd <- ggplot(dd) +
  geom_col(aes(xpos, yval, fill = col), width = 0.34, alpha = 0.85, colour = "white") +
  geom_hline(yintercept = 0, linewidth = 0.36) +
  geom_text(data = data.frame(x = 1:2 - 0.18, z = z_vals),
            aes(x, ifelse(z >= 0, z + 2.0, 2.0), label = sprintf("%.1f", z)),
            colour = MARDIA_COL, fontface = "bold", size = 2.82) +
  scale_fill_identity() +
  scale_x_continuous(breaks = 1:2, labels = labk) +
  scale_y_continuous(name = "Mardia z", sec.axis = sec_axis(~ z_to_e(.), name = "99th-pct excess")) +
  coord_cartesian(ylim = c(zlo, zhi)) +
  labs(x = NULL, tag = "d") +
  theme_all() +
  theme(axis.title.y.left = element_text(colour = MARDIA_COL),
        axis.text.y.left = element_text(colour = MARDIA_COL),
        axis.title.y.right = element_text(colour = EXCESS_COL),
        axis.text.y.right = element_text(colour = EXCESS_COL))

# (e) Layer-2 cohort — domains carrying individual deviation (|z|>2)
prof <- read.delim(PROFILE_TSV, sep = "\t", check.names = FALSE)
zcols <- grep("^z_", names(prof), value = TRUE)
frac <- vapply(zcols, function(cc) mean(abs(prof[[cc]]) > 2) * 100, numeric(1))
names(frac) <- sub("^z_", "", zcols)
od  <- order(frac)
ids <- names(frac)[od]
ddv <- data.frame(name = factor(dlabel(ids), levels = dlabel(ids)),
                  val = frac[od], grp = dgroup(ids))
gauss_row <- which(levels(ddv$name) == "Test scores")
if (length(gauss_row) == 0) gauss_row <- 2
p_dev <- ggplot(ddv, aes(val, name, fill = grp)) +
  geom_col(colour = "white", width = 0.78) +
  geom_vline(xintercept = 4.55, linetype = "dotted", colour = "#333", linewidth = 0.4) +
  annotate("text", x = 4.7, y = gauss_row, label = "Gaussian 4.6%", size = 2.82, hjust = 0, colour = "#333") +
  scale_fill_manual(values = GROUP_FILL, name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.02))) +
  labs(x = "% of patients with |deviation z| > 2", y = NULL, tag = "e") +
  theme_all() + theme(legend.position = c(0.77, 0.20), legend.text = element_text(size = 8),
                     legend.key.size = unit(0.30, "cm"), axis.text.y = element_text(size = 8),
                     plot.margin = margin(t = 4, r = 4, b = 2, l = 2))

# assemble: row1 (a,b) | row2 (c,d) | row3 (e)
r1 <- pa + pb + plot_layout(widths = c(1, 1))
r2 <- pc_hist + pd + plot_layout(widths = c(1, 1))
fig <- r1 / r2 / free(p_dev, type = "panel", side = "l") + plot_layout(heights = c(1, 1, 1.15))

save_fig(fig, "Figure5", width = 7.0866, height = 7.2)   # 180 mm wide
