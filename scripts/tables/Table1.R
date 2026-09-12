#!/usr/bin/env Rscript
# Table 1 — dataset / cohort summary (489 reports, A/P x autism/non-autism + unique children)
rm(list = ls())
source("_common.R")
stopifnot(file.exists(META))

m <- read.csv(META, stringsAsFactors = FALSE)
m$type  <- ifelse(grepl("-A[0-9]+$", m$code), "A",
            ifelse(grepl("-P[0-9]+$", m$code), "P", NA))
m$dx    <- ifelse(m$final_diag == 1, "Autism", "Non-autism")
m$child <- sub("^(KHU-[A-Z]-[0-9]+).*$", "\\1", m$code)

cnt <- function(dx, ty) sum(m$dx == dx & m$type == ty)
report_tot <- function(ty) sum(m$type == ty)
uchild <- function(ty) length(unique(m$child[m$type == ty]))

t1 <- data.frame(
  Group = c("Autism", "Non-autism", "Report total", "Unique individuals"),
  `A-type` = c(cnt("Autism", "A"), cnt("Non-autism", "A"), report_tot("A"), uchild("A")),
  `P-type` = c(cnt("Autism", "P"), cnt("Non-autism", "P"), report_tot("P"), uchild("P")),
  Total = c(sum(m$dx == "Autism"), sum(m$dx == "Non-autism"), nrow(m), length(unique(m$child))),
  check.names = FALSE,
  stringsAsFactors = FALSE)
write_table(t1, "Table1")
