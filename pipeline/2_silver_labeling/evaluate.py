from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from schemas import DomainDef, DomainRegistry, EvalMetrics, EvalReport, GoldLabel, SentenceLabel

logger = logging.getLogger(__name__)

def _resolve_to_domain_id(
    val: Any,
    registry: Optional[DomainRegistry],
) -> Optional[str]:

    label_str = str(val)
    if registry and label_str in registry.code_to_id:

        return registry.code_to_id[label_str]
    if registry and label_str in registry.id_to_code:

        return label_str
    if registry:
        logger.warning("No registered domain for code/ID %s; using it as-is", val)

    return label_str

def load_gold_labels(
    path: str,
    registry: Optional[DomainRegistry] = None,
) -> Dict[Tuple[str, int], Set[str]]:

    gold: Dict[Tuple[str, int], Set[str]] = {}
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Gold label file not found: {path}")

    with open(p, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                gl = GoldLabel(**data)
                key = (gl.report_id, gl.sentence_idx)

                domain_ids: Set[str] = set()
                for label_val in gl.labels:
                    did = _resolve_to_domain_id(label_val, registry)
                    if did is not None:
                        domain_ids.add(did)

                gold.setdefault(key, set()).update(domain_ids)
            except Exception as e:
                logger.warning("Failed to parse gold label (line %d): %s", line_no, e)

    logger.info("Loaded gold labels for %d sentences", len(gold))
    return gold

def load_silver_labels(path: str) -> Dict[Tuple[str, int], Set[str]]:

    silver: Dict[Tuple[str, int], Set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                key = (data["report_id"], data["sentence_idx"])
                domain_ids: Set[str] = set()
                labels_data = data.get("labels")
                if isinstance(labels_data, list):
                    for item in labels_data:
                        if isinstance(item, dict):
                            did = item.get("domain_id")
                            if did:
                                domain_ids.add(did)
                else:

                    label_data = data.get("label")
                    if isinstance(label_data, dict):
                        did = label_data.get("domain_id")
                        if did:
                            domain_ids.add(did)
                if not domain_ids:
                    domain_ids.add("other_general")
                silver.setdefault(key, set()).update(domain_ids)
            except Exception as e:
                logger.debug("Skipping malformed silver label: %s", e)

    return silver

def compute_metrics(
    gold: Dict[Tuple[str, int], Set[str]],
    silver: Dict[Tuple[str, int], Set[str]],
    all_domain_ids: List[str],
    threshold: float,
    model: str,
) -> EvalReport:

    common_keys = set(gold.keys()) & set(silver.keys())
    if not common_keys:
        logger.warning("No matched sentences between gold and silver labels")
        return EvalReport(
            per_domain=[],
            accuracy=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            micro_precision=0.0,
            micro_recall=0.0,
            micro_f1=0.0,
            subset_accuracy=0.0,
            hamming_loss=0.0,
            total_sentences=0,
            threshold_used=threshold,
            model=model,
        )

    logger.info("Evaluation sentences: %d / gold %d / silver %d",
                len(common_keys), len(gold), len(silver))

    domain_set = set(all_domain_ids)

    tp: Dict[str, int] = {d: 0 for d in all_domain_ids}
    fp: Dict[str, int] = {d: 0 for d in all_domain_ids}
    fn: Dict[str, int] = {d: 0 for d in all_domain_ids}
    support: Dict[str, int] = {d: 0 for d in all_domain_ids}

    subset_correct = 0
    hamming_mismatch = 0

    confusion: Dict[str, Dict[str, int]] = {
        g: {s: 0 for s in all_domain_ids} for g in all_domain_ids
    }

    for key in common_keys:
        g_set = gold[key]
        p_set = silver[key]

        if g_set == p_set:
            subset_correct += 1

        for d in all_domain_ids:
            in_g = d in g_set
            in_p = d in p_set
            if in_g:
                support[d] += 1
            if in_g and in_p:
                tp[d] += 1
            elif in_p and not in_g:
                fp[d] += 1
            elif in_g and not in_p:
                fn[d] += 1

            if in_g != in_p:
                hamming_mismatch += 1

        for g in g_set & domain_set:
            for s in p_set & domain_set:
                confusion[g][s] += 1

    subset_accuracy = subset_correct / len(common_keys) if common_keys else 0.0

    total_cells = len(common_keys) * len(all_domain_ids)
    hamming_loss = hamming_mismatch / total_cells if total_cells > 0 else 0.0

    per_domain: List[EvalMetrics] = []
    for d in all_domain_ids:
        p = tp[d] / (tp[d] + fp[d]) if (tp[d] + fp[d]) > 0 else 0.0
        r = tp[d] / (tp[d] + fn[d]) if (tp[d] + fn[d]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_domain.append(EvalMetrics(
            domain_id=d,
            precision=round(p, 4),
            recall=round(r, 4),
            f1=round(f1, 4),
            support=support[d],
        ))

    active = [m for m in per_domain if m.support > 0]
    macro_p = sum(m.precision for m in active) / len(active) if active else 0.0
    macro_r = sum(m.recall for m in active) / len(active) if active else 0.0
    macro_f1 = sum(m.f1 for m in active) / len(active) if active else 0.0

    total_support = sum(m.support for m in per_domain)
    weighted_f1 = (
        sum(m.f1 * m.support for m in per_domain) / total_support
        if total_support > 0
        else 0.0
    )

    sum_tp = sum(tp.values())
    sum_fp = sum(fp.values())
    sum_fn = sum(fn.values())
    micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0
        else 0.0
    )

    return EvalReport(
        per_domain=per_domain,
        accuracy=round(subset_accuracy, 4),
        macro_precision=round(macro_p, 4),
        macro_recall=round(macro_r, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        micro_precision=round(micro_p, 4),
        micro_recall=round(micro_r, 4),
        micro_f1=round(micro_f1, 4),
        subset_accuracy=round(subset_accuracy, 4),
        hamming_loss=round(hamming_loss, 4),
        total_sentences=len(common_keys),
        threshold_used=threshold,
        model=model,
        confusion_matrix=confusion,
    )

def save_eval_report(report: EvalReport, output_dir: str) -> Tuple[str, str]:

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out / "eval_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)

    csv_path = out / "eval_report.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain_id", "precision", "recall", "f1", "support"])
        for m in report.per_domain:
            writer.writerow([m.domain_id, m.precision, m.recall, m.f1, m.support])
        writer.writerow([])
        writer.writerow(["SUBSET_ACCURACY", "", "", report.subset_accuracy,
                         report.total_sentences])
        writer.writerow(["HAMMING_LOSS", "", "", report.hamming_loss, ""])
        writer.writerow(["MACRO", report.macro_precision, report.macro_recall,
                         report.macro_f1, ""])
        writer.writerow(["MICRO", report.micro_precision, report.micro_recall,
                         report.micro_f1, ""])
        writer.writerow(["WEIGHTED_F1", "", "", report.weighted_f1, ""])

    logger.info("Saved evaluation report: %s, %s", json_path, csv_path)
    return str(json_path), str(csv_path)

# ── CLI ──────────────────────────────────────────────────────

def run_evaluation(config: Dict[str, Any]) -> EvalReport:

    gold_path = config["paths"].get("gold_labels")
    if not gold_path:
        raise ValueError("config.paths.gold_labels is not set.")

    silver_path = Path(config["paths"]["output_dir"]) / "silver_labels.jsonl"
    if not silver_path.exists():
        raise FileNotFoundError(f"Silver label file not found: {silver_path}")

    domains = [
        DomainDef(
            code=str(d.get("code", "")),
            id=d["id"],
            name_ko=d.get("name_ko", ""),
            name_en=d.get("name_en", ""),
            description=d.get("description", ""),
            pilot_active=d.get("pilot_active", True),
        )
        for d in config["domains"]
    ]
    registry = DomainRegistry(domains)

    gold = load_gold_labels(gold_path, registry=registry)
    silver = load_silver_labels(str(silver_path))

    all_domain_ids = [d["id"] for d in config["domains"]]

    report = compute_metrics(
        gold=gold,
        silver=silver,
        all_domain_ids=all_domain_ids,
        threshold=config["threshold"]["confidence_min"],
        model=config["llm"]["model"],
    )

    save_eval_report(report, config["paths"]["output_dir"])

    print("\n" + "=" * 72)
    print("  Evaluation Summary (Multi-label)")
    print("=" * 72)
    print(f"  Evaluated sentences: {report.total_sentences}")
    print(f"  Subset Accuracy:  {report.subset_accuracy:.4f}")
    print(f"  Hamming Loss:     {report.hamming_loss:.4f}")
    print(f"  Macro P/R/F1:     {report.macro_precision:.4f} / "
          f"{report.macro_recall:.4f} / {report.macro_f1:.4f}")
    print(f"  Micro P/R/F1:     {report.micro_precision:.4f} / "
          f"{report.micro_recall:.4f} / {report.micro_f1:.4f}")
    print(f"  Weighted F1:      {report.weighted_f1:.4f}")
    print("-" * 72)
    print(f"  {'Code':>4} {'Domain':<30} {'P':>6} {'R':>6} {'F1':>6} {'N':>5}")
    print("-" * 72)
    for m in report.per_domain:
        code = registry.id_to_code.get(m.domain_id, "?")
        print(f"  {code:>4} {m.domain_id:<30} {m.precision:>6.3f} {m.recall:>6.3f} "
              f"{m.f1:>6.3f} {m.support:>5}")
    print("=" * 72)

    return report
