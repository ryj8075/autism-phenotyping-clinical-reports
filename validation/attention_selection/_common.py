#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Paths
HERE = Path(__file__).resolve().parent
OUTPUTS_DIR = HERE / "outputs"
FIGURES_DIR = HERE / "figures"

REPO_ROOT = HERE.parents[1]
REPO_PIPELINE = REPO_ROOT / "pipeline"

PATHS: Dict[str, Path] = {
    # Silver labels: sentence-level, 489 reports, the production domain vectors are built from.
    "silver_labels": REPO_PIPELINE
    / "2_silver_labeling/data/silver_label/silver_labels_489reports.jsonl",
    # Expert Gold labels: sentence-level, 26 reports
    "gold_labels": REPO_PIPELINE
    / "2_silver_labeling/data/gold_label/gold_labels_26reports.jsonl",
    # Attention intermediates (489 reports)
    "intermediates": REPO_PIPELINE
    / "1_classifier/intermediates/489samples_epoch40_153stc_128tkn_epoch40_patience10_no_headings",
    # Domain list / definitions (19 domains)
    "domain_meta": REPO_PIPELINE
    / "3_multidomain_vectors/domain_frequency_vector/outputs/domain_vectors_meta_latest.json",
    # Diagnosis metadata fallback
    "metadata": REPO_PIPELINE / "1_classifier/data/metadata/metadata_489reports.csv",
}

# Production default top-K
TOP_K_PRIMARY = 10
K_SWEEP = [5, 10, 15, 20]
N_RANDOM_DRAWS = 200
RANDOM_SEED = 42

# Domain list
def load_domains() -> Tuple[List[str], List[str], Dict[str, str], Dict[str, str]]:
    with open(PATHS["domain_meta"], encoding="utf-8") as f:
        meta = json.load(f)

    domain_order = list(meta["domain_columns"])
    defs = {d["id"]: d for d in meta["domain_definitions"]}

    codes = [defs[d]["code"] for d in domain_order]
    id_to_code = {d: defs[d]["code"] for d in domain_order}
    id_to_group = {d: defs[d]["code"][:2] for d in domain_order}  # group prefix CO/AS/RE
    return domain_order, codes, id_to_code, id_to_group

# Label loaders
def _normalise_label_items(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen = set()
    for item in raw:
        did: Optional[str] = None
        if isinstance(item, str):
            did = item
        elif isinstance(item, dict):
            did = item.get("domain_id")
        if not did or did in seen:
            continue
        seen.add(did)
        out.append(did)
    return out

def load_sentence_labels(jsonl_path: Path) -> Dict[Tuple[str, int], List[str]]:
    out: Dict[Tuple[str, int], List[str]] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            key = (str(data["report_id"]), int(data["sentence_idx"]))
            raw = data.get("labels")
            if raw is None:
                single = data.get("label")
                raw = [single] if isinstance(single, dict) else []
            out[key] = _normalise_label_items(raw)
    return out

def report_to_sentence_indices(
    sentence_labels: Dict[Tuple[str, int], List[str]],
) -> Dict[str, List[int]]:
    """Group available sentence_idx values per report_id (sorted)."""
    by_report: Dict[str, List[int]] = {}
    for (rid, sidx) in sentence_labels.keys():
        by_report.setdefault(rid, []).append(sidx)
    for rid in by_report:
        by_report[rid].sort()
    return by_report

# Domain-vector builder
def build_vector(
    sentences_labels: List[List[str]],
    domain_order: List[str],
) -> Optional[np.ndarray]:

    idx = {d: i for i, d in enumerate(domain_order)}
    mass = np.zeros(len(domain_order), dtype=float)

    for labels in sentences_labels:
        active = [d for d in labels if d in idx]
        if not active:
            continue  # skip sentence with no usable label
        w = 1.0 / len(active)
        for d in active:
            mass[idx[d]] += w

    total = mass.sum()
    if total <= 0:
        return None
    return mass / total

# Attention loader
class AttentionStore:
    def __init__(self) -> None:
        d = PATHS["intermediates"]
        self.attn = np.load(d / "attention_matrices_np.npy")          # (N,S,S)
        self.valid_mask = np.load(d / "valid_sentence_mask_np.npy")   # (N,S) bool
        self.n_valid = np.load(d / "n_valid_sentences_np.npy")        # (N,)
        self.report_ids = np.array(
            [str(x) for x in np.load(d / "report_id_array.npy", allow_pickle=True)]
        )
        self.labels = np.load(d / "labels_np.npy").astype(int)        # (N,) diagnosis 0/1
        self.rid_to_row = {rid: i for i, rid in enumerate(self.report_ids)}

    def importance(self, report_id: str) -> np.ndarray:
        """Column-sum importance over valid sentences; invalid set to -inf."""
        i = self.rid_to_row[report_id]
        cs = self.attn[i].sum(axis=0)            # column_sum (key importance)
        mask = self.valid_mask[i]
        out = np.where(mask, cs, -np.inf)
        return out

    def valid_indices(self, report_id: str) -> np.ndarray:
        i = self.rid_to_row[report_id]
        return np.flatnonzero(self.valid_mask[i])

    def top_k_indices(self, report_id: str, k: int) -> List[int]:
        """Top-k valid sentence indices by column-sum importance (rank order)."""
        imp = self.importance(report_id)
        valid = self.valid_indices(report_id)
        kk = min(k, len(valid))
        order = np.argsort(imp)[::-1][:kk]
        return [int(x) for x in order]

    def diagnosis(self, report_id: str) -> int:
        return int(self.labels[self.rid_to_row[report_id]])

# Matplotlib style
def apply_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,  # editable text in PDF
            "ps.fonttype": 42,
        }
    )

# Color scheme for domain groups (CO Core ASD / AS associated / RE report elements)
GROUP_COLORS = {
    "CO": "#C0392B",  # Core ASD domains
    "AS": "#2E86C1",  # Associated and co-occurring features
    "RE": "#7F8C8D",  # Report elements and other
}

def save_fig(fig, name: str) -> None:
    """Save a figure as both PNG (300 dpi) and PDF with a fixed name."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")

def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
