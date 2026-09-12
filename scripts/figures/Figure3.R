#!/usr/bin/env Rscript
# Fig 3 — per-domain silver-vs-gold reliability lollipop (489/19).
rm(list = ls())
source("_common.R")

SILVER <- file.path(FS_DIR, "silver_labels_489reports.jsonl")
GOLD   <- file.path(FS_DIR, "gold_labels_26reports.jsonl")

dom <- domain_columns()
ordered_codes <- c(paste0("CO", 1:7), paste0("AS", 1:9), paste0("RE", seq_len(length(dom) - 16)))
stopifnot(length(dom) == length(ordered_codes))

read_jsonl <- function(p) lapply(readLines(p, warn = FALSE), function(l) if (nzchar(trimws(l))) fromJSON(l) else NULL)
silver_rows <- Filter(Negate(is.null), read_jsonl(SILVER))
gold_rows   <- Filter(Negate(is.null), read_jsonl(GOLD))
gold_reports <- sort(unique(vapply(gold_rows, function(r) r$report_id, character(1))))
zero_mat <- function(reps) matrix(0, length(reps), length(dom), dimnames = list(reps, dom))
Scnt <- zero_mat(gold_reports); Gcnt <- zero_mat(gold_reports)
get_domains <- function(labels) {
  if (is.null(labels) || length(labels) == 0) return(character(0))
  if (is.data.frame(labels)) return(as.character(labels$domain_id))
  as.character(unlist(labels))
}
for (r in silver_rows) { rid <- r$report_id; if (!(rid %in% gold_reports)) next
  for (dl in get_domains(r$labels)) if (dl %in% dom) Scnt[rid, dl] <- Scnt[rid, dl] + 1 }
for (r in gold_rows) { rid <- r$report_id
  for (dl in get_domains(r$labels)) if (dl %in% dom) Gcnt[rid, dl] <- Gcnt[rid, dl] + 1 }
S <- Scnt / pmax(rowSums(Scnt), 1)
G <- Gcnt / pmax(rowSums(Gcnt), 1)

# per-domain Spearman rho (silver vs gold proportion across the 26 gold reports)
rho <- vapply(dom, function(d) {
  s <- S[, d]; g <- G[, d]
  if (sd(s) < 1e-10 || sd(g) < 1e-10) NA_real_ else suppressWarnings(cor(s, g, method = "spearman"))
}, numeric(1))

# unified domain display names (sentence case; shared map in _common.R)
name <- dlabel(dom)

map <- data.frame(code = ordered_codes, name = name, r = as.numeric(rho[dom]), stringsAsFactors = FALSE)
map$group <- factor(substr(map$code, 1, 2), levels = c("CO", "AS", "RE"),
                    labels = c("Core ASD\ndomains",
                               "Associated and\nco-occurring\nfeatures",
                               "Report\nelements\nand other"))
map$label <- factor(paste(map$code, map$name), levels = rev(paste(map$code, map$name)))
map$rv    <- ifelse(is.na(map$r), 0, map$r)
map$txt   <- ifelse(is.na(map$r), "n/a", sprintf("%.2f", map$r))

# RdBu diverging endpoints (ColorBrewer 11-class): blue(neg) - white(0) - red(pos)
BLUE <- "#35618F"; MIDW <- "#F7F7F7"; RED <- "#C15B4E"

p <- ggplot(map, aes(x = rv, y = label)) +
  geom_vline(xintercept = 0, colour = "#c2c8d2", linewidth = 0.4) +
  geom_vline(xintercept = c(0.60, 0.80), linetype = "22", colour = "#b8bfc9", linewidth = 0.36) +
  geom_segment(aes(x = 0, xend = rv, yend = label, colour = r), linewidth = 1.1, lineend = "round") +
  geom_point(aes(fill = r), shape = 21, size = 3.0, stroke = 0.4, colour = "grey35") +
  geom_text(aes(label = txt, hjust = ifelse(rv < 0, 1.3, -0.3)), size = 2.82,
            colour = "#2b3a52", fontface = "bold") +
  facet_grid(rows = vars(group), scales = "free_y", space = "free_y", switch = "y") +
  scale_fill_gradient2(name = "Spearman ρ", low = BLUE, mid = MIDW, high = RED, midpoint = 0,
                       limits = c(-0.5, 1.0), breaks = c(-0.5, 0, 0.5, 1.0), na.value = "#c9cdd4") +
  scale_colour_gradient2(low = BLUE, mid = MIDW, high = RED, midpoint = 0,
                         limits = c(-0.5, 1.0), na.value = "#c9cdd4", guide = "none") +
  scale_x_continuous(limits = c(-0.75, 1.15), breaks = c(-0.5, 0, 0.5, 1.0)) +
  labs(x = "silver-vs-gold proportion   Spearman ρ   (n = 26 gold reports)", y = NULL) +
  guides(fill = guide_colourbar(barwidth = unit(30, "mm"), barheight = unit(3, "mm"),
                                title.position = "left", title.vjust = 1)) +
  theme_all(8) +
  theme(legend.position = "top", legend.justification = "center",
        strip.placement = "outside",
        strip.text.y.left = element_text(angle = 90, face = "bold", size = 8, colour = "#16233d"),
        axis.text.y = element_text(size = 8, colour = "#1d2b40"),
        axis.title.x = element_text(size = 8, margin = margin(t = 5)),
        axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        plot.margin = margin(4, 8, 4, 4))

save_fig(p, "Figure3", width = 7.0866, height = 7.6)
