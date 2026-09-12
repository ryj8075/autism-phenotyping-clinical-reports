#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

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

    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("output_dir", "outputs")

    return cfg

def decode_sentences_from_tensor(
    input_tensor: torch.Tensor,
    sample_idx: int,
    tokenizer,
) -> List[str]:

    report = input_tensor[sample_idx]  # (num_sentences, seq_length)
    sentences = []
    for sent_tokens in report:
        tokens = sent_tokens[sent_tokens != tokenizer.pad_token_id]
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        sentences.append(text)
    return sentences

def compute_attention_importance(
    attention_matrix: np.ndarray,
    method: str = "column_sum",
) -> np.ndarray:

    if method == "column_sum":
        return attention_matrix.sum(axis=0)
    elif method == "row_sum":
        return attention_matrix.sum(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-extract top-attention sentences from all reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_all_high_attention_sentences.py
  python extract_all_high_attention_sentences.py --top_k 10
  python extract_all_high_attention_sentences.py --attention_method row_sum
        """,
    )
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Main configuration file (default: config.yaml)")
    parser.add_argument("--top_k", "-k", type=int, default=None,
                        help="Number of top-attention sentences (overrides config)")
    parser.add_argument("--attention_method", choices=["column_sum", "row_sum"],
                        default=None, help="Attention aggregation method")
    parser.add_argument("--intermediates_dir", type=str, default=None)
    parser.add_argument("--tokenized_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config = load_config(str(config_path))

    if args.top_k is not None:
        config["analysis"]["top_k"] = args.top_k
    if args.attention_method is not None:
        config["analysis"]["attention_method"] = args.attention_method
    if args.intermediates_dir is not None:
        config["paths"]["intermediates_dir"] = args.intermediates_dir
    if args.tokenized_dir is not None:
        config["paths"]["tokenized_dir"] = args.tokenized_dir
    if args.output_dir is not None:
        config["paths"]["output_dir"] = args.output_dir

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    top_k = config["analysis"]["top_k"]
    method = config["analysis"]["attention_method"]
    config_dir = config_path.parent
    output_dir = Path(_resolve_path(config_dir, config["paths"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    nlp_root = Path(_resolve_path(config_dir, config["paths"]["nlp_asd_root"]))
    intermediates_dir = config["paths"].get("intermediates_dir")
    tokenized_dir = config["paths"].get("tokenized_dir")

    if not intermediates_dir:
        logger.error("config.paths.intermediates_dir is not set.")
        sys.exit(1)
    if not tokenized_dir:
        logger.error("config.paths.tokenized_dir is not set.")
        sys.exit(1)

    if not Path(intermediates_dir).is_absolute():
        intermediates_dir = _resolve_path(config_dir, intermediates_dir)
    if not Path(tokenized_dir).is_absolute():
        tokenized_dir = _resolve_path(config_dir, tokenized_dir)

    # Attention matrices
    logger.info("Loading attention matrices...")
    attn_path = os.path.join(intermediates_dir, "attention_matrices_np.npy")
    if not os.path.exists(attn_path):
        logger.error("Attention matrix file not found: %s", attn_path)
        sys.exit(1)
    attention_matrices = np.load(attn_path)
    N = attention_matrices.shape[0]
    logger.info("Attention matrices: shape=%s", attention_matrices.shape)

    # Labels
    labels_path = os.path.join(intermediates_dir, "labels_np.npy")
    if os.path.exists(labels_path):
        labels_np = np.load(labels_path)
    else:
        labels_np = torch.load(
            os.path.join(tokenized_dir, "label_tensor"),
            map_location="cpu", weights_only=False,
        ).numpy()

    # Report IDs
    report_ids_path = os.path.join(intermediates_dir, "report_ids.npy")
    if os.path.exists(report_ids_path):
        report_id_array = np.load(report_ids_path, allow_pickle=True)
    else:
        report_id_array = torch.load(
            os.path.join(tokenized_dir, "report_id_array"),
            map_location="cpu", weights_only=False,
        )
        if isinstance(report_id_array, torch.Tensor):
            report_id_array = report_id_array.numpy()

    logger.info("Loading tokenized tensor: %s", tokenized_dir)
    input_tensor = torch.load(
        os.path.join(tokenized_dir, "input_tensor"),
        map_location="cpu", weights_only=False,
    )

    # Tokenizer
    tokenizer_name = config.get("model", {}).get("tokenizer_name", "klue/roberta-base")
    logger.info("Loading tokenizer: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    logger.info("Loaded %d reports", N)

    logger.info("Extracting top-%d sentences from %d reports...", top_k, N)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    records = []

    for i in range(N):
        report_id = str(report_id_array[i])
        asd_label = int(labels_np[i])
        attn_matrix = attention_matrices[i]  # (S, S)

        # Attention importance
        importance = compute_attention_importance(attn_matrix, method)

        k = min(top_k, len(importance))
        top_indices = np.argsort(importance)[::-1][:k]

        sentences = decode_sentences_from_tensor(input_tensor, i, tokenizer)

        sent_records = []
        for rank, idx in enumerate(top_indices):
            idx = int(idx)
            text = sentences[idx] if idx < len(sentences) else ""
            sent_records.append({
                "rank": rank + 1,
                "sentence_idx": idx,
                "attention_score": round(float(importance[idx]), 6),
                "text": text,
            })

        records.append({
            "report_id": report_id,
            "asd_label": asd_label,
            "total_sentences": len(sentences),
            "sentences": sent_records,
        })

        if (i + 1) % 50 == 0 or i == N - 1:
            logger.info("  processed: %d / %d", i + 1, N)

    jsonl_ts = output_dir / f"high_attention_sentences_{timestamp}.jsonl"
    jsonl_latest = output_dir / "high_attention_sentences_latest.jsonl"

    for path in [jsonl_ts, jsonl_latest]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Saved JSONL: %s", jsonl_latest)

    tsv_rows = []
    for rec in records:
        for s in rec["sentences"]:
            tsv_rows.append({
                "report_id": rec["report_id"],
                "asd_label": rec["asd_label"],
                "rank": s["rank"],
                "sentence_idx": s["sentence_idx"],
                "attention_score": s["attention_score"],
                "text": s["text"],
            })

    df_tsv = pd.DataFrame(tsv_rows)
    tsv_ts = output_dir / f"high_attention_sentences_{timestamp}.tsv"
    tsv_latest = output_dir / "high_attention_sentences_latest.tsv"
    df_tsv.to_csv(tsv_ts, sep="\t", index=False)
    df_tsv.to_csv(tsv_latest, sep="\t", index=False)

    logger.info("Saved TSV: %s", tsv_latest)

    meta = {
        "timestamp": timestamp,
        "top_k": top_k,
        "attention_method": method,
        "n_reports": N,
        "intermediates_dir": intermediates_dir,
        "tokenized_dir": tokenized_dir,
    }
    meta_ts = output_dir / f"high_attention_meta_{timestamp}.json"
    meta_latest = output_dir / "high_attention_meta_latest.json"
    for path in [meta_ts, meta_latest]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("  High-attention sentence extraction complete")
    print("=" * 72)
    print(f"  Reports:         {N}")
    print(f"  Top-K:           {top_k}")
    print(f"  Attention method:{method}")
    print(f"  JSONL:           {jsonl_latest}")
    print(f"  TSV:             {tsv_latest}")
    print("=" * 72)

if __name__ == "__main__":
    main()
