#!/usr/bin/env python3
"""Slim-data generator for figS1 panel c (report-embedding PCA).
Reads the 489-report attention-weighted sentence embeddings (460 MB intermediate),
builds one report-level embedding = mean over each report's valid sentences, runs a
2D PCA, attaches the MD-assessed diagnosis, and writes a tiny CSV that figS4.R reads.
Re-run only when the model run changes.  Output: figure_source/figS1_embedding_pca.csv"""
from pathlib import Path
import csv
import os
import re
import numpy as np
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[3]
INT = Path(os.environ.get(
    "CLASSIFIER_INTERMEDIATES_DIR",
    REPO_ROOT / "pipeline" / "1_classifier" / "intermediates" /
    "489samples_epoch40_153stc_128tkn_epoch40_patience10_no_headings",
))
META = Path(os.environ.get(
    "CLASSIFIER_METADATA_CSV",
    REPO_ROOT / "pipeline" / "1_classifier" / "data" / "metadata" /
    "metadata_489reports.csv",
))
OUT  = Path(__file__).resolve().parent.parent / "figure_source" / "figS1_embedding_pca.csv"

emb  = np.load(INT / "attention_weighted_sentence_embs_np.npy")   # (489, 153, 768)
mask = np.load(INT / "valid_sentence_mask_np.npy")                # (489, 153) bool
rid  = np.load(INT / "report_id_array.npy").astype(str)

rep = np.vstack([emb[i][mask[i]].mean(0) if mask[i].sum() else emb[i, 0] for i in range(emb.shape[0])])
pca = PCA(n_components=2).fit(rep)
xy  = pca.transform(rep)
evr = pca.explained_variance_ratio_ * 100

dx, dx_patient = {}, {}
REPORT_SUFFIX_RE = re.compile(r"(?:-\d+)?-[AP]\d+$")

def patient_id(code):
    return REPORT_SUFFIX_RE.sub("", str(code))

for r in csv.DictReader(open(META)):
    lab = "Autism" if r["final_diag"] == "1" else "Non-autism"
    dx[r["code"]] = lab
    dx_patient[patient_id(r["code"])] = lab

def diag(code):
    if code in dx:
        return dx[code]
    return dx_patient.get(patient_id(code), "NA")

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["report_id", "PC1", "PC2", "diagnosis", "pc1_var_pct", "pc2_var_pct"])
    for i, r in enumerate(rid):
        w.writerow([r, round(float(xy[i, 0]), 4), round(float(xy[i, 1]), 4),
                    diag(r), round(float(evr[0]), 1), round(float(evr[1]), 1)])
print(f"wrote {OUT}  (n={len(rid)}, PC1 {evr[0]:.1f}% / PC2 {evr[1]:.1f}%)")
