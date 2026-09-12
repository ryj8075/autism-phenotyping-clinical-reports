from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# 1. Silver labels
def load_silver_labels(
    silver_path: str | Path,
) -> Dict[Tuple[str, int], List[Tuple[str, float]]]:

    silver_path = Path(silver_path)
    labels: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}
    with open(silver_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                key = (str(data["report_id"]), int(data["sentence_idx"]))
            except (json.JSONDecodeError, KeyError):
                continue

            raw = data.get("labels")
            if raw is None:
                single = data.get("label")
                raw = [single] if isinstance(single, dict) else []
            if not isinstance(raw, list):
                raw = []

            seen: Dict[str, float] = {}
            order: List[str] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                did = item.get("domain_id")
                if not did:
                    continue
                conf = float(item.get("confidence") or 0.0)
                if did in seen:
                    seen[did] = max(seen[did], conf)
                else:
                    seen[did] = conf
                    order.append(did)

            labels[key] = [(d, seen[d]) for d in order] if order else [("other_general", 0.0)]
    return labels

# 2. Domain list
def load_active_domains(domains_yaml: str | Path,
                        pilot_mode: Optional[bool] = None) -> List[str]:

    path = Path(domains_yaml)
    with open(path, "r", encoding="utf-8") as f:
        dcfg = yaml.safe_load(f)
    if pilot_mode is None:
        pilot_mode = dcfg.get("pilot_mode", False)

    all_domains: List[dict] = []
    for category in dcfg.get("categories", []):
        for subcat in category.get("subcategories", []):
            for domain in subcat.get("domains", []):
                all_domains.append(domain)

    if pilot_mode:
        active = [d for d in all_domains if d.get("pilot_active", True)]
    else:
        active = all_domains
    return [d["id"] for d in active]

# 3. Top-K selection from attention matrix
def extract_top_k_indices(
    attention_matrices: np.ndarray,  # (N, S, S)
    top_k: int,
    method: str = "column_sum",
    valid_mask: Optional[np.ndarray] = None,  # (N, S) True=valid
) -> np.ndarray:

    N, S, _ = attention_matrices.shape
    top_k = int(min(top_k, S))

    if method == "column_sum":
        importance = attention_matrices.sum(axis=1)  # (N, S)
    elif method == "row_sum":
        importance = attention_matrices.sum(axis=2)
    else:
        raise ValueError(f"Unknown attention method: {method}")

    if valid_mask is not None:
        importance = np.where(valid_mask, importance, -np.inf)

    top_idx = np.argsort(-importance, axis=1)[:, :top_k]
    return top_idx


def weighted_sample_indices(
    attention_matrices: np.ndarray,  # (N, S, S)
    sample_size: int,
    rng: np.random.Generator,
    method: str = "column_sum",
    valid_mask: Optional[np.ndarray] = None,
    replace: bool = False,
) -> np.ndarray:

    N, S, _ = attention_matrices.shape

    if method == "column_sum":
        importance = attention_matrices.sum(axis=1)
    elif method == "row_sum":
        importance = attention_matrices.sum(axis=2)
    else:
        raise ValueError(f"Unknown method: {method}")

    importance = np.maximum(importance, 0)
    if valid_mask is not None:
        importance = importance * valid_mask.astype(float)

    out = np.zeros((N, sample_size), dtype=int)
    for i in range(N):
        w = importance[i]
        total = w.sum()
        if total <= 0:

            candidates = np.where(valid_mask[i])[0] if valid_mask is not None else np.arange(S)
            if len(candidates) == 0:
                out[i, :] = 0
                continue
            size_eff = min(sample_size, len(candidates)) if not replace else sample_size
            chosen = rng.choice(candidates, size=size_eff, replace=replace)
        else:
            p = w / total

            if valid_mask is not None:
                valid_idx = np.where(valid_mask[i])[0]
            else:
                valid_idx = np.arange(S)
            valid_p = p[valid_idx]
            valid_p = valid_p / valid_p.sum() if valid_p.sum() > 0 else None

            size_eff = min(sample_size, len(valid_idx)) if not replace else sample_size
            chosen = rng.choice(valid_idx, size=size_eff, replace=replace, p=valid_p)

        if len(chosen) < sample_size:

            pad = np.full(sample_size - len(chosen), chosen[-1] if len(chosen) else 0)
            chosen = np.concatenate([chosen, pad])
        out[i, :] = chosen
    return out

# 5. Domain vector construction
def build_proportion_matrix(
    indices: np.ndarray,  # (N, K)
    report_id_array: Sequence[str],
    silver_labels: Dict[Tuple[str, int], List[Tuple[str, float]]],
    active_domains: List[str],
    allocation: str = "equal",
) -> Tuple[np.ndarray, Dict]:

    N, K = indices.shape
    D = len(active_domains)
    domain_to_idx = {d: j for j, d in enumerate(active_domains)}

    mass = np.zeros((N, D), dtype=np.float64)
    matched_sentences = 0
    no_silver = 0
    no_mapped = 0
    total_selected = 0
    label_assignments = 0
    unmapped_assignments = 0

    for i in range(N):
        rid = str(report_id_array[i])
        for j in range(K):
            sidx = int(indices[i, j])
            total_selected += 1
            doms = silver_labels.get((rid, sidx))
            if doms is None:
                no_silver += 1
                continue
            m_dom = [(d, c) for (d, c) in doms if d in domain_to_idx]
            unmapped_assignments += len(doms) - len(m_dom)
            if not m_dom:
                no_mapped += 1
                continue
            matched_sentences += 1
            if allocation == "confidence":
                tot = sum(c for _, c in m_dom)
                weights = [c / tot for _, c in m_dom] if tot > 0 else [1.0 / len(m_dom)] * len(m_dom)
            else:
                weights = [1.0 / len(m_dom)] * len(m_dom)
            for (d, _), w in zip(m_dom, weights):
                mass[i, domain_to_idx[d]] += w
                label_assignments += 1

    row_sums = mass.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    proportion = mass / safe_sums

    stats = {
        "N": N,
        "K": K,
        "D": D,
        "allocation": allocation,
        "total_selected": int(total_selected),
        "matched": int(matched_sentences),
        "unmatched_no_silver_label": int(no_silver),
        "no_mapped_domain": int(no_mapped),
        "label_assignments": int(label_assignments),
        "unmapped_domain": int(unmapped_assignments),

        "match_rate": float(matched_sentences / total_selected) if total_selected > 0 else 0.0,
        "zero_rows": int((mass.sum(axis=1) == 0).sum()),
    }
    return proportion, stats

# 6. Valid sentence mask
def compute_valid_sentence_mask(attn_mask_tensor: np.ndarray) -> np.ndarray:
    valid_token_count = attn_mask_tensor.sum(axis=-1)
    return valid_token_count > 2
