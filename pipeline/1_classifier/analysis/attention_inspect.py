import os
import sys
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analyze_phenotypes_ko import analyze_high_attention_sentences

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data")
INTERMEDIATES_PATH = os.path.join(ROOT, "intermediates")
BASE_MODEL_NAME = "klue/roberta-base"

def decode_sentences_from_tensor(input_tensor, sample_idx, tokenizer):

    report = input_tensor[sample_idx]  # (num_sentences, seq_length)
    sentences = []
    for sent_tokens in report:
        tokens = sent_tokens[sent_tokens != tokenizer.pad_token_id]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        sentences.append(text.strip())
    return sentences

def inspect_sample(attns, sentences, sample_idx, label_str, threshold=0.1, top_k=10):

    res = analyze_high_attention_sentences(
        attns, sentences, sample_idx, label_str, threshold=threshold, top_k=top_k
    )

    res['matrix_stats'] = {
        'shape': tuple(int(s) for s in attns.shape),
        'max_value': float(np.max(attns)),
        'min_value': float(np.min(attns)),
        'mean_value': float(np.mean(attns)),
        'max_position': res['max_position'],
    }

    query_attention_sum = attns.sum(axis=1)
    q_ranking_indices = np.argsort(query_attention_sum)[::-1][:top_k]
    res['query_sentence_ranking'] = []
    for rank, idx in enumerate(q_ranking_indices):
        sent = sentences[idx] if idx < len(sentences) and sentences[idx] else "[PADDING]"
        res['query_sentence_ranking'].append({
            'rank': rank + 1,
            'sentence_idx': int(idx),
            'total_attention': float(query_attention_sum[idx]),
            'max_attention': float(np.max(attns[idx, :])),
            'sentence': sent,
        })

    return res

def print_results(results, sample_info):
    print("=" * 80)
    print("ATTENTION WEIGHT ANALYSIS")
    print("=" * 80)
    print("\nSample information:")
    for k, v in sample_info.items():
        print(f"  {k}: {v}")

    s = results['matrix_stats']
    print("\n--- Attention matrix statistics ---")
    print(f"  Shape: {s['shape']}")
    print(f"  Max: {s['max_value']:.6f}  Min: {s['min_value']:.6f}  Mean: {s['mean_value']:.6f}")
    print(f"  Max position: Query {s['max_position'][0]}, Key {s['max_position'][1]}")

    print("\n--- High attention pairs (top 15) ---")
    if results['high_attention_pairs']:
        for i, p in enumerate(results['high_attention_pairs'][:15]):
            print(f"\n  [{i+1}] Query {p['query_idx']} -> Key {p['key_idx']}: {p['attention_weight']:.6f}")
            print(f"      Key sentence: {p['key_sentence'][:120]}")
    else:
        print("  No pairs above threshold")

    print("\n--- Key sentence ranking (column sum, top 5) ---")
    for it in results['key_sentence_ranking'][:5]:
        print(f"  #{it['rank']} [idx {it['sentence_idx']}] total={it['total_attention']:.4f} "
              f"max={it['max_attention']:.6f}")
        print(f"      {it['sentence'][:120]}")

    print("\n--- Query sentence ranking (row sum, top 5) ---")
    for it in results['query_sentence_ranking'][:5]:
        print(f"  #{it['rank']} [idx {it['sentence_idx']}] total={it['total_attention']:.4f} "
              f"max={it['max_attention']:.6f}")
        print(f"      {it['sentence'][:120]}")
    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Inspect attention for a single sample")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample_idx", type=int, help="Sample index to inspect (0-indexed)")
    group.add_argument("--report_id", type=str, help="Report ID to inspect, e.g. REPORT-075-A01")

    parser.add_argument("--experiment_name", required=True, type=str,
                        help="Experiment name under intermediates/")
    parser.add_argument("--threshold", type=float, default=0.1, help="Attention weight threshold [default: 0.1]")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top items [default: 10]")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--tokenized_path", type=str, default=None,
                        help="Tokenized tensor directory [default: data/reports_tokenized]")
    args = parser.parse_args()

    inter = os.path.join(INTERMEDIATES_PATH, args.experiment_name)
    if not os.path.isdir(inter):
        print(f"[ERROR] Missing intermediates directory: {inter}")
        sys.exit(1)

    attention_matrices = np.load(os.path.join(inter, "attention_matrices_np.npy"))
    labels = np.load(os.path.join(inter, "labels_np.npy"))

    tokenized_path = args.tokenized_path or os.path.join(DATA_PATH, "reports_tokenized")
    if not os.path.isdir(tokenized_path):
        print(f"[ERROR] Missing tokenized data directory: {tokenized_path}")
        sys.exit(1)

    input_tensor = torch.load(os.path.join(tokenized_path, "input_tensor"),
                              map_location=torch.device('cpu'), weights_only=False)
    report_id_array = torch.load(os.path.join(tokenized_path, "report_id_array"),
                                 map_location=torch.device('cpu'), weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    report_id_list = [str(rid) for rid in report_id_array]
    if args.sample_idx is not None:
        sample_idx = args.sample_idx
        report_id = report_id_list[sample_idx] if sample_idx < len(report_id_list) else f"Sample_{sample_idx}"
    else:
        if args.report_id in report_id_list:
            sample_idx = report_id_list.index(args.report_id)
        else:
            matches = [i for i, rid in enumerate(report_id_list) if args.report_id in rid]
            if not matches:
                print(f"[ERROR] report_id '{args.report_id}' not found")
                sys.exit(1)
            sample_idx = matches[0]
        report_id = args.report_id

    if sample_idx >= len(attention_matrices):
        print(f"[ERROR] sample_idx {sample_idx} is out of range (n={len(attention_matrices)})")
        sys.exit(1)

    sentences = decode_sentences_from_tensor(input_tensor, sample_idx, tokenizer)
    label_str = "ASD" if labels[sample_idx] == 1 else "Non-ASD"

    results = inspect_sample(
        attention_matrices[sample_idx], sentences, sample_idx, label_str,
        threshold=args.threshold, top_k=args.top_k
    )

    sample_info = {
        'Sample Index': sample_idx,
        'Report ID': report_id,
        'Label': label_str,
        'Total Sentences': len(sentences),
        'Threshold': args.threshold,
    }
    print_results(results, sample_info)

    if args.output:
        import json

        def conv(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({'sample_info': sample_info, 'results': results},
                      f, ensure_ascii=False, indent=2, default=conv)
        print(f"\nSaved results: {args.output}")

if __name__ == "__main__":
    main()
