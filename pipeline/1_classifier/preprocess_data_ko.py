import os
import sys
import argparse
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.hierarchical_tokenizer_ko import (
    hierarchical_tokenizer_korean
)

def main():
    parser = argparse.ArgumentParser(description="Preprocess Korean clinical report data")

    parser.add_argument("--input_type", type=str, required=True, choices=["csv", "txt"],
                        help="Input data format: 'csv' or 'txt'")

    parser.add_argument("--csv_path", type=str, default=None,
                        help="CSV file path when input_type='csv'")
    parser.add_argument("--text_col", type=str, default="text",
                        help="Text column name")
    parser.add_argument("--label_col", type=str, default="final_diag",
                        help="Label column name")
    parser.add_argument("--report_id_col", type=str, default="report_id",
                        help="Report ID column name")
    parser.add_argument("--pat_id_col", type=str, default="pat_id",
                        help="Patient ID column name")

    parser.add_argument("--txt_dir", type=str, default=None,
                        help="Text file directory when input_type='txt'")
    parser.add_argument("--meta_path", type=str, default=None,
                        help="Metadata CSV path when input_type='txt'")

    parser.add_argument("--model_name", type=str, default="klue/roberta-base",
                        help="HuggingFace tokenizer/model name")
    parser.add_argument("--sentence_max_length", type=int, default=64,
                        help="Maximum tokens per sentence")
    parser.add_argument("--sentence_min_length", type=int, default=30,
                        help="Minimum sentence length in characters")
    parser.add_argument("--report_max_length", type=int, default=64,
                        help="Maximum sentences per report")

    parser.add_argument("--output_dir", type=str, default="data/reports_tokenized",
                        help="Output directory")

    args = parser.parse_args()

    print("=" * 50)
    print("Korean clinical report preprocessing")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Maximum tokens per sentence: {args.sentence_max_length}")
    print(f"Maximum sentences per report: {args.report_max_length}")

    if args.input_type == "csv":
        raise NotImplementedError(
            "CSV input is not implemented in this public pipeline. "
            "Use --input_type txt with --txt_dir and --meta_path."
        )

    elif args.input_type == "txt":
        if args.txt_dir is None or args.meta_path is None:
            raise ValueError("When input_type='txt', set both --txt_dir and --meta_path.")

        print(f"\nText directory: {args.txt_dir}")
        print(f"Metadata: {args.meta_path}")

        report_files = []
        for root, dirs, files in os.walk(args.txt_dir):
            for file in files:
                if file.endswith(".txt"):
                    report_files.append(os.path.join(root, file))

        print(f"Discovered text files: {len(report_files)}")

        meta_frame = pd.read_csv(args.meta_path)

        input_tensor, attention_mask_tensor, label_tensor, report_id_array = hierarchical_tokenizer_korean(
            report_files=report_files,
            meta_frame=meta_frame,
            tokenizer_name=args.model_name,
            sentence_max_length=args.sentence_max_length,
            sentence_min_length=args.sentence_min_length,
            report_max_length=args.report_max_length
        )

    print(f"\n=== Preprocessing results ===")
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Attention mask shape: {attention_mask_tensor.shape}")
    print(f"Label tensor shape: {label_tensor.shape}")
    print(f"Report IDs: {len(report_id_array)}")
    print(f"Label distribution: positive={label_tensor.sum().item()}, negative={len(label_tensor) - label_tensor.sum().item()}")

    os.makedirs(args.output_dir, exist_ok=True)

    torch.save(input_tensor, os.path.join(args.output_dir, "input_tensor"))
    torch.save(attention_mask_tensor, os.path.join(args.output_dir, "attention_mask_tensor"))
    torch.save(label_tensor, os.path.join(args.output_dir, "label_tensor"))
    torch.save(report_id_array, os.path.join(args.output_dir, "report_id_array"))

    print(f"\nSaved outputs: {args.output_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()
