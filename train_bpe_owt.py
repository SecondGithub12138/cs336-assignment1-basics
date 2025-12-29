#!/usr/bin/env python3
"""
Train a BPE tokenizer on OWT dataset using Hugging Face tokenizers (fast Rust implementation).
"""

import os
import time
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def main():
    print("=" * 80)
    print("Training BPE Tokenizer with Hugging Face (Rust)")
    print("=" * 80)

    # Configuration
    data_dir = Path("data")
    input_path = data_dir / "owt_train.txt"
    output_dir = Path("outputs") / "owt_tokenizer"
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 32_000
    special_tokens = ["<|endoftext|>"]

    # Check if input file exists
    if not input_path.exists():
        print(f"❌ Error: Training data not found at {input_path}")
        return

    # Print configuration
    file_size_mb = input_path.stat().st_size / 1024 / 1024
    print(f"\n📁 Configuration:")
    print(f"   Input file: {input_path}")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Vocab size: {vocab_size:,}")
    print(f"   Special tokens: {special_tokens}")
    print(f"   Output directory: {output_dir}")

    # Train BPE tokenizer
    print(f"\n🚀 Starting training...")
    start_time = time.time()

    # Create tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train
    tokenizer.train([str(input_path)], trainer)

    training_time = time.time() - start_time

    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    # Print stats
    print(f"\n✅ Training completed!")
    print(f"   Time: {training_time:.2f}s ({training_time / 60:.2f}m)")
    print(f"   Vocab size: {tokenizer.get_vocab_size():,}")
    print(f"   Saved to: {tokenizer_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
