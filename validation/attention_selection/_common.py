#!/usr/bin/env python3
"""
_common.py — Shared loaders, the domain-vector builder, and plotting helpers
for the attention/phenotype faithfulness validation sub-pipeline (1.7).

This module is read-only with respect to the rest of the repository: it only
LOADS existing artifacts (silver/gold labels, attention matrices, the domain
list). Nothing here writes into other pipeline folders.

Centralised paths live at the top of this file (PATHS). Every per-test script
imports from here so that path/logic changes happen in exactly one place.

Silver source: this sub-pipeline loads the CANONICAL silver
(silver_labels_26reports.jsonl, the full multi-label variant: other_general/C2 retained) -- the SAME file the production domain vectors
(3.2 outputs) are built from. The build_vector LOGIC below also matches production,
so these numbers validate the selector on the exact canonical phenotype space.

Key reused logic (computation kept faithful to production):
  - Per-sentence attention score = column_sum of the attention matrix,
    restricted to valid sentences. This matches
    pipeline/3_multidomain_vectors/top_10_sentences/extract_all_high_attention_sentences.py
    (compute_attention_importance, method="column_sum"), and because invalid
    columns sum to exactly 0 the production full-argsort naturally lands on
    valid sentences. We additionally mask invalid positions for robustness.
  - build_vector(): each sentence distributes mass 1.0 EQUALLY across its
    (active) silver labels, masses are summed over the sentence set, then the
    vector is L1-normalised to proportions (sum=1). This mirrors
    pipeline/3_multidomain_vectors/domain_frequency_vector/build_domain_vectors.py
    (allocation="equal", proportion vector). Sentences with no usable label are
    skipped.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ===========================================================================
# Centralised paths (the ONLY place paths are defined)
# ===========================================================================

HERE = Path(__file__).resolve().parent
OUTPUTS_DIR = HERE / "outputs"
FIGURES_DIR = HERE / "figures"

REPO_ROOT = HERE.parents[1]
REPO_PIPELINE = REPO_ROOT / "pipeline"

PATHS: Dict[str, Path] = {
    # Silver labels: CANONICAL silver (silver_labels_26reports.jsonl = full multi-label variant,
    # other_general/C2 retained) -- the SAME file
    # the production domain vectors (3.2 outputs) are built from.
    "silver_labels": REPO_PIPELINE
    / "2_silver_labeling/data/silver_label/silver_labels_26reports.jsonl",
    # Expert GOLD labels: sentence-level, 26 reports
    "gold_labels": REPO_PIPELINE
    / "2_silver_labeling/data/gold_label/gold_labels_26reports.jsonl",
    # Attention intermediates (489 reports)
    "intermediates": REPO_PIPELINE
    / "1_classifier/intermediates/489samples_epoch40_153stc_128tkn_epoch40_patience10_no_headings",
    # Domain list / definitions (19 domains, data-driven)
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

# ===========================================================================
# Domain list (data-driven; NOT hardcoded)
# ===========================================================================

def load_domains() -> Tuple[List[str], List[str], Dict[str, str], Dict[str, str]]:
    """
    Load the 19-domain order and code mapping from the production meta JSON.

    Returns
    -------
    domain_order : list[str]      domain_id in canonical column order
    codes        : list[str]      domain_code aligned with domain_order (A1..C3)
    id_to_code   : dict
    id_to_group  : dict           domain_id -> "A" | "B" | "C"
    """
    with open(PATHS["domain_meta"], encoding="utf-8") as f:
        meta = json.load(f)

    domain_order = list(meta["domain_columns"])
    defs = {d["id"]: d for d in meta["domain_definitions"]}

    codes = [defs[d]["code"] for d in domain_order]
    id_to_code = {d: defs[d]["code"] for d in domain_order}
    id_to_group = {d: defs[d]["code"][0] for d in domain_order}  # first char A/B/C
    return domain_order, codes, id_to_code, id_to_group

# ===========================================================================
# Label loaders
# ===========================================================================

def _normalise_label_items(raw: Any) -> List[str]:
    """
    Reduce a 'labels' field to a list of domain_id strings, de-duplicated,
    order preserved. Handles BOTH schemas:
      - silver: [{"domain_id": ..., "confidence": ...}, ...]
      - gold:   ["other_general", "social_emotional_reciprocity", ...]
      - legacy single 'label' dict is handled by the caller.
    """
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
    """
    Load a sentence-level label file (silver OR gold) into:
        (report_id, sentence_idx) -> [domain_id, ...]

    Sentences with an empty/absent label list map to [] (caller decides to skip).
    """
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

# ===========================================================================
# Domain-vector builder
# ===========================================================================

def build_vector(
    sentences_labels: List[List[str]],
    domain_order: List[str],
) -> Optional[np.ndarray]:
    """
    Build a |domain_order|-dim L1-normalised domain frequency vector from a SET
    of sentences (each given as a list of domain_id labels).

    Mass rule (allocation="equal"): each sentence carries mass 1.0 split equally
    across its labels that are in domain_order; masses are summed per domain;
    the result is L1-normalised to proportions (sum=1).

    Sentences with no usable label (empty, or none in domain_order) are skipped.

    Returns
    -------
    np.ndarray (len domain_order,) summing to 1, OR None if no sentence
    contributed any mass (degenerate report -> caller decides).
    """
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

# ===========================================================================
# Attention loader (faithful to extract_all_high_attention_sentences.py)
# ===========================================================================

class AttentionStore:
    """
    Holds the per-report attention column-sum importance and valid masks.

    Per-sentence attention score = attention_matrix.sum(axis=0) (column_sum),
    restricted to valid sentences (invalid columns are 0 anyway). This is the
    exact production "high_attention" definition.
    """

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

# ===========================================================================
# Matplotlib style (project rule: no titles, Arial, light/no grid, png+pdf)
# ===========================================================================

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

# Color scheme for domain groups (ASD-core A / general B / formal C)
GROUP_COLORS = {
    "A": "#C0392B",  # ASD-core
    "B": "#2E86C1",  # general psychiatric
    "C": "#7F8C8D",  # formal / other
}

def save_fig(fig, name: str) -> None:
    """Save a figure as both PNG (300 dpi) and PDF with a fixed name."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")

def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
