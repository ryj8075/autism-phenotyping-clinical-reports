#!/usr/bin/env Rscript
# Supplementary Data 1 — per-report two-layer fingerprint (346 reports x Mahalanobis + 19 z)
rm(list = ls())
source("_common.R")
stopifnot(file.exists(P_PROF))

prof <- read.delim(P_PROF, check.names = FALSE, stringsAsFactors = FALSE)
names(prof)[names(prof) == "mahalanobis_full"]  <- "prototypicality_mahalanobis_full"
zc  <- grep("^z_", names(prof), value = TRUE)
ord <- c("report_id", "report_type", "prototypicality_mahalanobis_full",
         "max_dev_domain", "max_dev_z", zc)
prof <- prof[, intersect(ord, names(prof))]
num  <- vapply(prof, is.numeric, logical(1)); prof[num] <- lapply(prof[num], function(x) round(x, 3))

# name the domains as Table 1 does, in the max-deviation column and the z headers
prof$max_dev_domain <- unname(display_of[prof$max_dev_domain])
names(prof)[match(zc, names(prof))] <- vapply(sub("^z_", "", zc),
  function(d) sprintf("%s %s (z)", code_of[[d]], display_of[[d]]), character(1))
write_table(prof, "SupplementaryData1")
