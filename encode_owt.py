#!/usr/bin/env python3
"""
Encode text files using Hugging Face tokenizer and save as binary format for training.
"""

import os
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer


def encode_file(tokenizer_path, input_path, output_path, chunk_size=1024*1024, batch_size=10000):
    """
    Encode a text file using tokenizer and save as int32 binary.

    Args:
        tokenizer_path: Path to tokenizer.json
        input_path: Path to input text file
        output_path: Path to output .bin file
        chunk_size: Size of text chunks to read (chars)
        batch_size: Number of tokens to batch before writing
    """
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    print(f"Encoding {input_path}...")
    file_size_mb = Path(input_path).stat().st_size / 1024 / 1024
    print(f"  Input size: {file_size_mb:.2f} MB")

    total_tokens = 0
    with open(output_path, "wb") as f_out:
        with open(input_path, "r", encoding="utf-8") as f_in:
            batch = []
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break

                # Encode chunk
                encoding = tokenizer.encode(chunk)
                batch.extend(encoding.ids)

                # Write batch if large enough
                while len(batch) >= batch_size:
                    np.array(batch[:batch_size], dtype=np.int32).tofile(f_out)
                    total_tokens += batch_size
                    batch = batch[batch_size:]

                    if total_tokens % 1_000_000 == 0:
                        print(f"  Encoded {total_tokens:,} tokens...")

            # Write remaining
            if batch:
                np.array(batch, dtype=np.int32).tofile(f_out)
                total_tokens += len(batch)

    output_size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"  Output size: {output_size_mb:.2f} MB")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Saved to: {output_path}")


def main():
    print("=" * 80)
    print("Encoding OWT Dataset")
    print("=" * 80)

    # Paths
    tokenizer_path = Path("outputs/owt_tokenizer/tokenizer.json")
    data_dir = Path("data")
    output_dir = Path("encode")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encode train and valid
    files = [
        # ("owt_train.txt", "owt_train.bin"),
        ("owt_valid.txt", "owt_valid.bin"),
    ]

    for input_file, output_file in files:
        input_path = data_dir / input_file
        output_path = output_dir / output_file

        if not input_path.exists():
            print(f"\n⚠️  Skipping {input_file} (not found)")
            continue

        print(f"\n{'='*80}")
        encode_file(tokenizer_path, input_path, output_path)

    print("\n" + "=" * 80)
    print("✅ Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
