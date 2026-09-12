#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

def _resolve_path(base_dir: Path, value: str | Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else base_dir / path)

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    analysis = cfg.setdefault("analysis", {})
    analysis.setdefault("top_k", 5)
    analysis.setdefault("attention_method", "column_sum")
    analysis.setdefault("exclude_other", False)
    analysis.setdefault("renormalize", True)
    analysis.setdefault("vector_types", ["proportion", "count", "binary"])
    analysis.setdefault("label_allocation", "equal")  # equal | confidence

    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("output_dir", "outputs")

    return cfg

def load_domains(domains_path: str, pilot_mode: Optional[bool] = None) -> Dict[str, Any]:
    with open(domains_path, encoding="utf-8") as f:
        dcfg = yaml.safe_load(f)

    if pilot_mode is None:
        pilot_mode = dcfg.get("pilot_mode", False)

    all_domains = []
    for category in dcfg.get("categories", []):
        cat_name = category.get("name", "")
        for subcat in category.get("subcategories", []):
            subcat_name = subcat.get("name", "")
            for domain in subcat.get("domains", []):
                domain["category"] = cat_name
                domain["subcategory"] = subcat_name
                all_domains.append(domain)

    if pilot_mode:
        active_domains = [d for d in all_domains if d.get("pilot_active", True)]
    else:
        active_domains = all_domains

    active_ids = [d["id"] for d in active_domains]

    logger.info(
        "Loaded domains: total=%d, active=%d (pilot_mode=%s)",
        len(all_domains), len(active_domains), pilot_mode,
    )

    return {
        "all_domains": all_domains,
        "active_domains": active_domains,
        "active_ids": active_ids,
        "pilot_mode": pilot_mode,
    }

def load_extracted_sentences(
    jsonl_path: str,
    top_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    report_ids: List[str] = []
    labels: List[int] = []
    all_indices: List[List[int]] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            report_ids.append(str(record["report_id"]))
            labels.append(int(record["asd_label"]))

            sentences = sorted(record.get("sentences", []), key=lambda s: s["rank"])
            if top_k is not None and top_k < len(sentences):
                sentences = sentences[:top_k]

            indices = [int(s["sentence_idx"]) for s in sentences]
            all_indices.append(indices)

    max_k = max(len(idx_list) for idx_list in all_indices) if all_indices else 0

    N = len(report_ids)
    top_k_indices = np.full((N, max_k), -1, dtype=int)
    for i, idx_list in enumerate(all_indices):
        top_k_indices[i, :len(idx_list)] = idx_list

    report_id_array = np.array(report_ids)
    labels_np = np.array(labels, dtype=int)

    logger.info(
        "Loaded JSONL: %d reports, max K=%d (requested top_k=%s)",
        N, max_k, top_k,
    )

    return report_id_array, labels_np, top_k_indices

def load_silver_labels(
    silver_path: str,
) -> Dict[Tuple[str, int], List[Tuple[str, float]]]:

    labels: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}
    n_lines = 0
    multi_label_lines = 0

    with open(silver_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.debug("Skipping malformed silver-label JSON: %s", e)
                continue

            key = (str(data["report_id"]), int(data["sentence_idx"]))

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

            if order:
                doms = [(d, seen[d]) for d in order]
                if len(doms) > 1:
                    multi_label_lines += 1
            else:
                doms = [("other_general", 0.0)]

            labels[key] = doms
            n_lines += 1

    logger.info(
        "Loaded silver labels for %d sentences (%d multi-label sentences)", n_lines, multi_label_lines
    )
    return labels

def _allocate_weights(
    mapped: List[Tuple[str, float]],
    allocation: str,
) -> List[float]:

    m = len(mapped)
    if allocation == "confidence":
        total = sum(c for _, c in mapped)
        if total > 0:
            return [c / total for _, c in mapped]

    return [1.0 / m] * m

def build_vectors(
    top_k_indices: np.ndarray,
    report_id_array: np.ndarray,
    silver_labels: Dict[Tuple[str, int], List[Tuple[str, float]]],
    active_ids: List[str],
    exclude_other: bool = False,
    other_domain_id: str = "other_general",
    renormalize: bool = True,
    allocation: str = "equal",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:

    N, K = top_k_indices.shape
    domain_list = list(active_ids)
    domain_to_idx = {d: i for i, d in enumerate(domain_list)}

    mass_matrix = np.zeros((N, len(domain_list)), dtype=float)

    total_selected = 0
    matched_sentences = 0
    no_silver_sentences = 0
    no_mapped_sentences = 0
    label_assignments = 0
    unmapped_assignments = 0
    per_report_details = []

    for i in range(N):
        report_id = str(report_id_array[i])
        report_domains: List[str] = []
        r_selected = r_matched = 0

        for j in range(K):
            sent_idx = int(top_k_indices[i, j])
            if sent_idx < 0:
                continue
            total_selected += 1
            r_selected += 1

            key = (report_id, sent_idx)
            doms = silver_labels.get(key)
            if doms is None:
                no_silver_sentences += 1
                continue

            mapped = [(d, c) for (d, c) in doms if d in domain_to_idx]
            unmapped_assignments += len(doms) - len(mapped)
            if not mapped:
                no_mapped_sentences += 1
                continue

            matched_sentences += 1
            r_matched += 1
            weights = _allocate_weights(mapped, allocation)
            for (d, _), w in zip(mapped, weights):
                mass_matrix[i, domain_to_idx[d]] += w
                report_domains.append(d)
                label_assignments += 1

        per_report_details.append({
            "report_id": report_id,
            "n_selected_sentences": r_selected,
            "matched_sentences": r_matched,
            "domains_found": report_domains,
        })

    proportion_computed = False

    if exclude_other and other_domain_id in domain_to_idx:
        other_idx = domain_to_idx[other_domain_id]
        keep_mask = np.ones(len(domain_list), dtype=bool)
        keep_mask[other_idx] = False

        if not renormalize:
            row_sums = mass_matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            proportion_matrix = (mass_matrix / row_sums)[:, keep_mask]
            proportion_computed = True

        mass_matrix = mass_matrix[:, keep_mask]
        domain_list = [d for d in domain_list if d != other_domain_id]

    if not proportion_computed:
        row_sums = mass_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        proportion_matrix = mass_matrix / row_sums

    binary_matrix = (mass_matrix > 0).astype(int)

    report_ids = [str(r) for r in report_id_array]

    df_count = pd.DataFrame(mass_matrix, index=report_ids, columns=domain_list)
    df_count.index.name = "report_id"

    df_proportion = pd.DataFrame(proportion_matrix, index=report_ids, columns=domain_list)
    df_proportion.index.name = "report_id"

    df_binary = pd.DataFrame(binary_matrix, index=report_ids, columns=domain_list)
    df_binary.index.name = "report_id"

    stats = {
        "allocation": allocation,
        "total_selected_sentences": total_selected,
        "matched_sentences": matched_sentences,
        "no_silver_label_sentences": no_silver_sentences,
        "no_mapped_domain_sentences": no_mapped_sentences,
        "sentence_match_rate": matched_sentences / max(1, total_selected),
        "label_assignments": label_assignments,
        "unmapped_label_assignments": unmapped_assignments,
        "per_report_details": per_report_details,
    }

    return df_proportion, df_count, df_binary, stats

def save_results(
    df_proportion: pd.DataFrame,
    df_count: pd.DataFrame,
    df_binary: pd.DataFrame,
    labels_np: np.ndarray,
    stats: Dict,
    config: Dict[str, Any],
    domain_info: Dict[str, Any],
    output_dir: str,
) -> None:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    for df in [df_proportion, df_count, df_binary]:
        if "asd_label" not in df.columns:
            df.insert(0, "asd_label", labels_np.astype(int))

    vector_types = config["analysis"]["vector_types"]

    if "proportion" in vector_types:
        path = out / f"domain_vectors_proportion_{timestamp}.tsv"
        df_proportion.to_csv(path, sep="\t", float_format="%.4f")
        df_proportion.to_csv(out / "domain_vectors_proportion_latest.tsv", sep="\t", float_format="%.4f")
        logger.info("Proportion TSV: %s", path)

    if "count" in vector_types:

        path = out / f"domain_vectors_count_{timestamp}.tsv"
        df_count.to_csv(path, sep="\t", float_format="%.4f")
        df_count.to_csv(out / "domain_vectors_count_latest.tsv", sep="\t", float_format="%.4f")
        logger.info("Count TSV: %s", path)

    if "binary" in vector_types:
        path = out / f"domain_vectors_binary_{timestamp}.tsv"
        df_binary.to_csv(path, sep="\t")
        df_binary.to_csv(out / "domain_vectors_binary_latest.tsv", sep="\t")
        logger.info("Binary TSV: %s", path)

    npy_data = df_proportion.drop(columns=["asd_label"]).values
    np.save(out / f"domain_vectors_proportion_{timestamp}.npy", npy_data)
    np.save(out / "domain_vectors_proportion_latest.npy", npy_data)

    domain_cols = [c for c in df_proportion.columns if c != "asd_label"]
    meta = {
        "timestamp": timestamp,
        "config": {
            "top_k": config["analysis"]["top_k"],
            "attention_method": config["analysis"]["attention_method"],
            "exclude_other": config["analysis"]["exclude_other"],
            "renormalize": config["analysis"]["renormalize"],
            "label_allocation": config["analysis"]["label_allocation"],
        },
        "pilot_mode": domain_info["pilot_mode"],
        "domain_columns": domain_cols,
        "n_reports": len(df_proportion),
        "n_domains": len(domain_cols),
        "domain_definitions": [
            {"code": d["code"], "id": d["id"], "name_ko": d["name_ko"]}
            for d in domain_info["active_domains"]
            if d["id"] in domain_cols
        ],
        "stats": {k: v for k, v in stats.items() if k != "per_report_details"},
    }
    with open(out / f"domain_vectors_meta_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(out / "domain_vectors_meta_latest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(out / f"matching_details_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(stats["per_report_details"], f, ensure_ascii=False, indent=2)
    with open(out / "matching_details_latest.json", "w", encoding="utf-8") as f:
        json.dump(stats["per_report_details"], f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("  Domain frequency vectors complete (v3, multi-label)")
    print("=" * 72)
    print(f"  Reports:        {meta['n_reports']}")
    print(f"  Active domains:   {meta['n_domains']}")
    print(f"  Pilot mode:      {meta['pilot_mode']}")
    print(f"  Top-K:            {meta['config']['top_k']}")
    print(f"  Allocation:        {stats['allocation']}")
    print(f"  Sentence match rate:      {stats['sentence_match_rate']:.1%}"
          f"  ({stats['matched_sentences']}/{stats['total_selected_sentences']})")
    print(f"    No silver label:    {stats['no_silver_label_sentences']}")
    print(f"    No active domain: {stats['no_mapped_domain_sentences']}")
    print(f"  Allocated label pairs:   {stats['label_assignments']}")
    print(f"  Discarded inactive labels:   {stats['unmapped_label_assignments']}")
    print("-" * 72)
    means = df_proportion[domain_cols].mean()
    print(f"  {'Code':<6} {'Domain':<35} {'Mean proportion':>10}")
    print("-" * 72)
    for d_def in meta["domain_definitions"]:
        did = d_def["id"]
        if did in means:
            print(f"  {d_def['code']:<6} {did:<35} {means[did]:>10.3f}")
    print("=" * 72)
    print(f"\n  Output: {output_dir}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build domain frequency vectors v3 (multi-label mass allocation x silver labels)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_domain_vectors.py                                      # default run (all 19 domains)
  python build_domain_vectors.py --top_k 10
  python build_domain_vectors.py --allocation confidence              # confidence-weighted allocation
  python build_domain_vectors.py --pilot_mode                         # pilot mode (A+C)
  python build_domain_vectors.py --exclude_other true                 # exclude other_general
  python build_domain_vectors.py --extracted_path my_sentences.jsonl
        """,
    )
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--domains", "-d", default=None,
                        help="Domain definition file (overrides config domains_file)")
    parser.add_argument("--top_k", "-k", type=int, default=None,
                        help="Number of top sentences to use from JSONL")
    parser.add_argument("--extracted_path", type=str, default=None,
                        help="Extracted sentence JSONL path (default: config.paths.extracted_path or {output_dir}/high_attention_sentences_latest.jsonl)")
    parser.add_argument("--allocation", choices=["equal", "confidence"], default=None,
                        help="Multi-label mass allocation method (default: config or equal)")
    parser.add_argument("--pilot_mode", action="store_true", default=False,
                        help="Enable pilot mode (pilot_active domains only)")
    parser.add_argument("--exclude_other", type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=None, help="Exclude other_general")
    parser.add_argument("--silver_labels_path", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config = load_config(str(config_path))

    domains_file = args.domains or config.get("domains_file", "domains.yaml")
    domains_path = Path(domains_file)
    if not domains_path.is_absolute():
        domains_path = Path(__file__).parent / domains_path
    pilot_mode = True if args.pilot_mode else None
    domain_info = load_domains(str(domains_path), pilot_mode=pilot_mode)

    if args.top_k is not None:
        config["analysis"]["top_k"] = args.top_k
    if args.exclude_other is not None:
        config["analysis"]["exclude_other"] = args.exclude_other
    if args.allocation is not None:
        config["analysis"]["label_allocation"] = args.allocation
    if args.silver_labels_path is not None:
        config["paths"]["silver_labels_path"] = args.silver_labels_path

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    config_dir = config_path.parent
    nlp_root = Path(_resolve_path(config_dir, config["paths"]["nlp_asd_root"]))
    output_dir = _resolve_path(config_dir, config["paths"]["output_dir"])
    silver_path = config["paths"].get("silver_labels_path")
    allocation = config["analysis"]["label_allocation"]

    if not silver_path:
        logger.error("config.paths.silver_labels_path is not set.")
        sys.exit(1)
    if not Path(silver_path).is_absolute():
        silver_path = _resolve_path(config_dir, silver_path)

    extracted_path = args.extracted_path or config["paths"].get("extracted_path")
    if extracted_path is None:
        extracted_path = os.path.join(output_dir, "high_attention_sentences_latest.jsonl")
    if not Path(extracted_path).is_absolute():
        extracted_path = _resolve_path(config_dir, extracted_path)

    if not os.path.exists(extracted_path):
        logger.error("Extracted sentence JSONL file not found: %s", extracted_path)
        logger.error("Run ../top_10_sentences/extract_all_high_attention_sentences.py first.")
        sys.exit(1)

    top_k = config["analysis"]["top_k"]

    logger.info("Config: top_k=%d, allocation=%s, pilot=%s",
                top_k, allocation, domain_info["pilot_mode"])
    logger.info("JSONL input: %s", extracted_path)
    logger.info("Silver labels: %s", silver_path)

    report_id_array, labels_np, top_k_indices = load_extracted_sentences(
        extracted_path, top_k=top_k,
    )
    logger.info("Loaded %d reports; top_k_indices shape=%s",
                len(report_id_array), top_k_indices.shape)

    silver_labels = load_silver_labels(silver_path)

    df_proportion, df_count, df_binary, stats = build_vectors(
        top_k_indices=top_k_indices,
        report_id_array=report_id_array,
        silver_labels=silver_labels,
        active_ids=domain_info["active_ids"],
        exclude_other=config["analysis"]["exclude_other"],
        other_domain_id="other_general",
        renormalize=config["analysis"]["renormalize"],
        allocation=allocation,
    )

    save_results(
        df_proportion=df_proportion,
        df_count=df_count,
        df_binary=df_binary,
        labels_np=labels_np,
        stats=stats,
        config=config,
        domain_info=domain_info,
        output_dir=output_dir,
    )

if __name__ == "__main__":
    main()
