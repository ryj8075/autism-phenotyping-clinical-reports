import re
import argparse
from pathlib import Path

def remove_headings(text: str) -> str:

    text = re.sub(r'## [^#]+? ##', '', text)

    text = re.sub(r' +', ' ', text)

    text = text.strip()

    return text

def process_file(input_path: Path, output_path: Path, encoding: str = "utf-8") -> None:

    with open(input_path, "r", encoding=encoding) as f:
        text = f.read()

    processed_text = remove_headings(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding=encoding) as f:
        f.write(processed_text)

    print(f"Processed: {input_path.name} -> {output_path}")

def process_directory(
    input_dir: str,
    output_dir: str,
    encoding: str = "utf-8",
    recursive: bool = True,
) -> None:

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if recursive:
        txt_files = list(input_path.rglob("*.txt"))
    else:
        txt_files = list(input_path.glob("*.txt"))

    if not txt_files:
        print(f"Warning: no text files found in {input_dir}")
        return

    print(f"Processing {len(txt_files)} text files...")

    for txt_file in sorted(txt_files):

        relative_path = txt_file.relative_to(input_path)
        out_file = output_path / relative_path

        try:
            process_file(txt_file, out_file, encoding)
        except Exception as e:
            print(f"Error processing {txt_file.name}: {e}")

    print(f"\nDone. Outputs saved to {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove headings formatted as ## ... ## from text files."
    )
    parser.add_argument(
        "input_dir",
        help="Input directory containing text files",
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_dir",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not include subdirectories",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding [default: utf-8]",
    )

    args = parser.parse_args()

    process_directory(
        args.input_dir,
        args.output_dir,
        encoding=args.encoding,
        recursive=not args.no_recursive,
    )
