#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import yaml

from evaluate import run_evaluation
from label import Labeler
from schemas import DomainDef, DomainRegistry, SentenceLabel
from segment import segment_sentences

def setup_logging(log_dir: str, verbose: bool = True) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"run_{ts}.log"

    handlers: list = [logging.FileHandler(log_file, encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=handlers,
    )

def load_domains_from_yaml(
    domains_path: str,
    pilot_mode: Optional[bool] = None,
) -> List[DomainDef]:

    with open(domains_path, encoding="utf-8") as f:
        dcfg = yaml.safe_load(f)

    if pilot_mode is None:
        pilot_mode = dcfg.get("pilot_mode", False)

    all_domains: List[DomainDef] = []
    for category in dcfg.get("categories", []):
        cat_name = category.get("name", "")
        for subcat in category.get("subcategories", []):
            subcat_name = subcat.get("name", "")
            for domain in subcat.get("domains", []):
                all_domains.append(DomainDef(
                    code=str(domain["code"]),
                    id=domain["id"],
                    name_ko=domain["name_ko"],
                    name_en=domain["name_en"],
                    description=domain.get("description", ""),
                    pilot_active=domain.get("pilot_active", True),
                    category=cat_name,
                    subcategory=subcat_name,
                ))

    if pilot_mode:
        active = [d for d in all_domains if d.pilot_active]
    else:
        active = all_domains

    logging.info(
        "Loaded domains: total=%d, active=%d (pilot_mode=%s, source=%s)",
        len(all_domains), len(active), pilot_mode, domains_path,
    )

    return active

_REQUIRED_KEYS = {
    "llm": dict,
    "prompt": dict,
    "threshold": dict,
    "paths": dict,
}
_REQUIRED_PATHS = ["input_dir", "output_dir", "log_dir"]
_REQUIRED_LLM = ["provider", "model"]

def _resolve_config_path(config_dir: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else config_dir / path)

def load_config(
    path: str,
    pilot_mode: Optional[bool] = None,
) -> Dict[str, Any]:

    config_dir = Path(path).resolve().parent

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key, expected_type in _REQUIRED_KEYS.items():
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")
        if not isinstance(cfg[key], expected_type):
            raise TypeError(
                f"config['{key}'] must be {expected_type.__name__}, "
                f"not {type(cfg[key]).__name__}."
            )

    for pk in _REQUIRED_PATHS:
        if pk not in cfg["paths"]:
            raise ValueError(f"Missing required config.paths key: {pk}")

    for lk in _REQUIRED_LLM:
        if lk not in cfg["llm"]:
            raise ValueError(f"Missing required config.llm key: {lk}")

    domains_file = cfg.get("domains_file")
    inline_domains = cfg.get("domains", [])

    if domains_file:

        domains_path = Path(domains_file)
        if not domains_path.is_absolute():
            domains_path = config_dir / domains_path
        if not domains_path.exists():
            raise FileNotFoundError(
                f"Domain file not found: {domains_path}\n"
                f"  Check config.domains_file."
            )
        domain_defs = load_domains_from_yaml(str(domains_path), pilot_mode=pilot_mode)

        cfg["domains"] = [d.model_dump() for d in domain_defs]
        logging.info("Using external domain file: %s (%d active)", domains_path, len(domain_defs))

    elif inline_domains:

        logging.info("Using inline domains: %d", len(inline_domains))

    else:
        raise ValueError(
            "No domains are defined.\n"
            "  Set config.domains_file or config.domains."
        )

    if not cfg["domains"]:
        raise ValueError("No active domains. Check pilot_mode.")

    for path_key in ["input_dir", "output_dir", "log_dir", "gold_labels"]:
        if path_key in cfg["paths"] and cfg["paths"][path_key]:
            cfg["paths"][path_key] = _resolve_config_path(config_dir, str(cfg["paths"][path_key]))

    return cfg

def load_progress(output_path: Path) -> set:

    done = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    done.add((data["report_id"], data["sentence_idx"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done

def run_label(config: Dict[str, Any], args: argparse.Namespace) -> None:

    input_dir = Path(config["paths"]["input_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "silver_labels.jsonl"

    seg_cfg = config.get("segmentation", {})
    runtime = config.get("runtime", {})
    resume = runtime.get("resume", True)
    save_every = runtime.get("save_every", 50)

    txt_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(".txt")
    ])

    if args.files:
        txt_files = [f for f in txt_files if f in args.files]

    if args.max_files:
        txt_files = txt_files[: args.max_files]

    if not txt_files:
        logging.error("No target files found.")
        return

    done = load_progress(output_path) if resume else set()
    if done:
        logging.info("Loaded progress: %d completed items", len(done))

    labeler = Labeler(config)

    all_tasks: List[tuple] = []  # (report_id, sentence_idx, sentence)
    for fname in txt_files:
        report_id = fname.replace(".txt", "")
        fpath = input_dir / fname
        with open(fpath, encoding="utf-8") as f:
            text = f.read()

        sentences = segment_sentences(
            text,
            method=seg_cfg.get("method", "regex"),
            min_length=seg_cfg.get("min_length", 10),
            max_length=seg_cfg.get("max_length", 500),
        )

        for idx, sent in enumerate(sentences):
            if (report_id, idx) not in done:
                all_tasks.append((report_id, idx, sent))

    total = len(all_tasks)
    logging.info(
        "Labeling targets: %d items (%d files, %d skipped)",
        total, len(txt_files), len(done),
    )

    if total == 0:
        logging.info("All sentences have already been labeled.")
        return

    try:
        from tqdm import tqdm
        task_iter = tqdm(all_tasks, desc="labeling", unit="sent")
    except ImportError:
        task_iter = all_tasks
        logging.info("tqdm is not installed; running without a progress bar.")

    buffer: List[str] = []
    processed = 0
    errors = 0
    t_start = time.time()

    try:
        for report_id, idx, sent in task_iter:
            try:
                result = labeler.label_sentence(sent, report_id, idx)
                line = result.model_dump_json(exclude_none=True)
                buffer.append(line)
                processed += 1

                if result.error:
                    errors += 1

            except Exception as e:
                logging.error("Fatal error [%s:%d]: %s", report_id, idx, e)
                err_record = SentenceLabel(
                    report_id=report_id,
                    sentence_idx=idx,
                    sentence=sent,
                    model=config["llm"]["model"],
                    error=str(e),
                )
                buffer.append(err_record.model_dump_json(exclude_none=True))
                errors += 1

            if len(buffer) >= save_every:
                _flush_buffer(buffer, output_path)
                buffer.clear()

    except KeyboardInterrupt:
        logging.warning("Interrupted by user (Ctrl+C); saving buffered records.")
    finally:

        if buffer:
            _flush_buffer(buffer, output_path)

    elapsed = time.time() - t_start
    logging.info(
        "Labeling complete: %d processed (%d errors), %.1f seconds elapsed (%.2f seconds/item)",
        processed, errors, elapsed,
        elapsed / max(1, processed),
    )

    registry = _make_registry(config)
    _generate_summary(output_path, output_dir, registry=registry)

def _make_registry(config: Dict[str, Any]) -> DomainRegistry:

    domains = []
    for d in config["domains"]:

        domains.append(DomainDef(
            code=str(d.get("code", "")),
            id=d["id"],
            name_ko=d.get("name_ko", ""),
            name_en=d.get("name_en", ""),
            description=d.get("description", ""),
            pilot_active=d.get("pilot_active", True),
            category=d.get("category", ""),
            subcategory=d.get("subcategory", ""),
        ))
    return DomainRegistry(domains)

def _flush_buffer(buffer: List[str], output_path: Path) -> None:

    with open(output_path, "a", encoding="utf-8") as f:
        for line in buffer:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    logging.debug("Intermediate save: %d records", len(buffer))

def _generate_summary(
    silver_path: Path,
    output_dir: Path,
    registry: Optional[DomainRegistry] = None,
) -> None:

    domain_counts: Counter = Counter()
    total_sentences = 0
    error_count = 0
    no_label_count = 0
    multilabel_count = 0
    total_labels = 0
    sentences_per_report: Counter = Counter()
    conf_sum: Dict[str, float] = {}
    conf_n: Dict[str, int] = {}

    with open(silver_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_sentences += 1
            rid = data.get("report_id", "?")
            sentences_per_report[rid] += 1

            if data.get("error"):
                error_count += 1
                continue

            labels = data.get("labels")
            if not labels or not isinstance(labels, list):
                no_label_count += 1
                continue

            n_valid = 0
            for label_data in labels:
                if not isinstance(label_data, dict):
                    continue
                did = label_data.get("domain_id")
                if not did:
                    continue
                domain_counts[did] += 1
                conf = label_data.get("confidence", 0)
                conf_sum[did] = conf_sum.get(did, 0) + conf
                conf_n[did] = conf_n.get(did, 0) + 1
                n_valid += 1

            if n_valid == 0:
                no_label_count += 1
            else:
                total_labels += n_valid
                if n_valid >= 2:
                    multilabel_count += 1

    labeled_sentences = total_sentences - error_count - no_label_count
    avg_labels = total_labels / max(1, labeled_sentences)

    csv_path = output_dir / "summary_report.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Item", "Value"])
        w.writerow(["Total sentences", total_sentences])
        w.writerow(["Error sentences", error_count])
        w.writerow(["No-label sentences", no_label_count])
        w.writerow(["Multi-label sentences (>=2)", multilabel_count])
        w.writerow(["Total labels", total_labels])
        w.writerow(["Mean labels per sentence", f"{avg_labels:.3f}"])
        w.writerow(["Reports", len(sentences_per_report)])
        w.writerow([])
        w.writerow(["Code", "Domain", "Count", "Percent(%)", "Mean confidence"])
        for did, cnt in domain_counts.most_common():
            code = registry.id_to_code.get(did, "?") if registry else "?"
            ratio = cnt / max(1, total_sentences) * 100
            avg_c = conf_sum.get(did, 0) / max(1, conf_n.get(did, 0))
            w.writerow([code, did, cnt, f"{ratio:.1f}", f"{avg_c:.3f}"])

    logging.info("Saved summary report: %s", csv_path)

    if registry:
        print("\n" + "=" * 72)
        print("  Domain Codebook")
        print("=" * 72)
        print(registry.summary_table())

    print("\n" + "=" * 72)
    print("  Labeling Summary Report (Multi-label)")
    print("=" * 72)
    print(f"  Total sentences:          {total_sentences}")
    print(f"  Errors:             {error_count}")
    print(f"  No labels:           {no_label_count}")
    print(f"  Multi-label(>=2):    {multilabel_count}")
    print(f"  Total labels:       {total_labels}")
    print(f"  Mean labels/sentence: {avg_labels:.3f}")
    print(f"  Reports:           {len(sentences_per_report)}")
    print("-" * 72)
    print(f"  {'Code':>4} {'Domain':<30} {'Count':>6} {'Percent':>7} {'avg_conf':>9}")
    print("-" * 72)
    for did, cnt in domain_counts.most_common():
        code = registry.id_to_code.get(did, "?") if registry else "?"
        ratio = cnt / max(1, total_sentences) * 100
        avg_c = conf_sum.get(did, 0) / max(1, conf_n.get(did, 0))
        print(f"  {code:>4} {did:<30} {cnt:>6} {ratio:>6.1f}% {avg_c:>9.3f}")
    print("=" * 72)

def run_summary(config: Dict[str, Any]) -> None:

    output_dir = Path(config["paths"]["output_dir"])

    silver_name = config["paths"].get("silver_labels_file", "silver_labels.jsonl")
    silver_path = output_dir / silver_name
    if not silver_path.exists():
        logging.error("Silver label file not found: %s", silver_path)
        return
    logging.info("Silver label file for summary: %s", silver_path)
    registry = _make_registry(config)
    _generate_summary(silver_path, output_dir, registry=registry)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ASD clinical report silver-labeling pipeline (multi-label, 19 domains)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py label                          # label with all 19 domains (default)
  python run.py label --pilot-mode             # pilot mode (A+C domains)
  python run.py label --max-files 3            # first 3 files only
  python run.py label --files REPORT-001-A01.txt REPORT-002-A01.txt
  python run.py evaluate                       # evaluate against gold labels
  python run.py summary                        # distribution summary report
  python run.py label --config my_config.yaml  # custom config
  python run.py label --domains-file /path/to/domains.yaml  # explicit domain file
  python run.py label --provider local --model llama3.1:8b --base-url http://localhost:11434/v1  # Ollama
  python run.py label --provider openai --model gpt-4o-mini   # OpenAI (switch via CLI without editing config)
        """,
    )
    parser.add_argument(
        "command",
        choices=["label", "evaluate", "summary"],
        help="Command to run",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )
    parser.add_argument(
        "--domains-file",
        default=None,
        help="Domain-definition YAML path (overrides config domains_file)",
    )
    parser.add_argument(
        "--pilot-mode",
        action="store_true",
        default=False,
        help="Enable pilot mode: use A+C domains only (default: all 19 domains)",
    )
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        default=None,
        help="Specific filenames to process (label command only)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (label command only)",
    )
    parser.add_argument(
        "--gold-labels",
        default=None,
        help="Gold-label JSONL path (overrides config for evaluate)",
    )

    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "local"],
        default=None,
        help="Override LLM provider (instead of config.llm.provider)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override LLM model (instead of config.llm.model), for example gpt-4o-mini or llama3.1:8b",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override LLM endpoint (instead of config.llm.base_url), such as an Ollama or vLLM URL for local provider",
    )
    return parser

# ── main ─────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    pilot_mode = True if args.pilot_mode else None

    config = load_config(str(config_path), pilot_mode=pilot_mode)

    if args.provider:
        config["llm"]["provider"] = args.provider
    if args.model:
        config["llm"]["model"] = args.model
    if args.base_url:
        config["llm"]["base_url"] = args.base_url

    if args.domains_file:
        domains_path = Path(args.domains_file)
        if not domains_path.is_absolute():
            domains_path = Path(__file__).parent / domains_path
        domain_defs = load_domains_from_yaml(str(domains_path), pilot_mode=pilot_mode)
        config["domains"] = [d.model_dump() for d in domain_defs]
        logging.info("CLI domain-file override: %s (%d active)", domains_path, len(domain_defs))

    setup_logging(
        config["paths"]["log_dir"],
        verbose=config.get("runtime", {}).get("verbose", True),
    )

    logging.info("Loaded config: %s", config_path)
    logging.info("Command: %s", args.command)
    logging.info("Active domains: %d", len(config["domains"]))
    _llm = config["llm"]
    logging.info(
        "LLM: provider=%s model=%s%s",
        _llm.get("provider"), _llm.get("model"),
        f" base_url={_llm['base_url']}" if _llm.get("base_url") else "",
    )

    if args.command == "label":
        run_label(config, args)
    elif args.command == "evaluate":

        if args.gold_labels:
            config["paths"]["gold_labels"] = args.gold_labels
        report = run_evaluation(config)

        print("-" * 72)
        print(f"  Micro F1:        {report.micro_f1:.4f}")
        print(f"  Subset accuracy: {report.subset_accuracy:.4f}")
        print(f"  Hamming loss:    {report.hamming_loss:.4f}")
        print("=" * 72)
    elif args.command == "summary":
        run_summary(config)

if __name__ == "__main__":
    main()
