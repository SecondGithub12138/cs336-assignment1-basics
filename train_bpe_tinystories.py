#!/usr/bin/env python3
"""
Train a byte-level BPE tokenizer on the TinyStories dataset.

This script trains a BPE tokenizer with vocab size 10,000 on the TinyStories
training data and saves the vocabulary and merges to disk.
"""

import os
import time
import json
import pickle
import psutil
from pathlib import Path
from tests.adapters import run_train_bpe


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def format_time(seconds):
    """Format seconds into human-readable string."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        return f"{secs:.1f}s"


def main():
    print("=" * 80)
    print("Training BPE Tokenizer on TinyStories Dataset")
    print("=" * 80)

    # Configuration
    data_dir = Path("data")
    input_path = data_dir / "TinyStoriesV2-GPT4-train.txt"

    # Extract base name from input file (e.g., "TinyStoriesV2-GPT4-valid")
    input_basename = input_path.stem

    output_dir = Path("outputs") / input_basename
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 32_000
    special_tokens = ["<|endoftext|>"]

    # Check if input file exists
    if not input_path.exists():
        print(f"❌ Error: Training data not found at {input_path}")
        print(f"   Please download TinyStories dataset first:")
        print(f"   mkdir -p data && cd data")
        print(f"   wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt")
        return

    # Print configuration
    file_size_mb = input_path.stat().st_size / 1024 / 1024
    print(f"\n📁 Configuration:")
    print(f"   Input file: {input_path}")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Vocab size: {vocab_size:,}")
    print(f"   Special tokens: {special_tokens}")
    print(f"   Output directory: {output_dir}")

    # Record initial memory
    initial_memory = get_memory_usage_mb()
    print(f"\n💾 Initial memory usage: {initial_memory:.2f} MB")

    # Train BPE tokenizer with detailed timing
    print(f"\n🚀 Starting training...")
    overall_start = time.time()

    # Start memory tracking
    import tracemalloc
    tracemalloc.start()

    vocab, merges = run_train_bpe(
        input_path=str(input_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    # Get peak memory from tracemalloc
    end_memory_bytes, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_memory_mb = end_memory_bytes / 1024 / 1024
    peak_memory_mb = peak_memory_bytes / 1024 / 1024

    overall_end = time.time()
    training_time = overall_end - overall_start

    # Record final memory (for comparison)
    final_memory_psutil = get_memory_usage_mb()

    # Print training stats
    print(f"\n✅ Training completed!")
    print(f"\n⏱️  Training time: {format_time(training_time)}")
    print(f"   ({training_time:.2f} seconds)")
    print(f"   ({training_time / 3600:.4f} hours)")

    print(f"\n💾 Memory usage (Python allocations tracked by tracemalloc):")
    print(f"   Peak:    {peak_memory_mb:.2f} MB")
    print(f"   Delta:   {peak_memory_mb - end_memory_mb:.2f} MB (freed)")
    print(f"\n   Note: Process RSS (psutil) = {final_memory_psutil:.2f} MB")
    print(f"         Initial RSS (psutil) = {initial_memory:.2f} MB")

    # Analyze vocabulary
    print(f"\n📊 Vocabulary statistics:")
    print(f"   Total tokens: {len(vocab):,}")
    print(f"   Number of merges: {len(merges):,}")

    # Find longest token
    longest_token = max(vocab.values(), key=len)
    longest_token_id = [k for k, v in vocab.items() if v == longest_token][0]

    print(f"\n🔍 Longest token:")
    print(f"   ID: {longest_token_id}")
    print(f"   Length: {len(longest_token)} bytes")
    print(f"   Bytes: {longest_token}")
    try:
        decoded = longest_token.decode('utf-8', errors='replace')
        print(f"   Text: '{decoded}'")
    except:
        print(f"   Text: (unable to decode)")

    # Show some example long tokens
    print(f"\n📝 Top 10 longest tokens:")
    sorted_tokens = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for i, (token_id, token_bytes) in enumerate(sorted_tokens, 1):
        try:
            decoded = token_bytes.decode('utf-8', errors='replace')
            print(f"   {i:2d}. ID={token_id:5d}, len={len(token_bytes):2d}, text='{decoded}'")
        except:
            print(f"   {i:2d}. ID={token_id:5d}, len={len(token_bytes):2d}, bytes={token_bytes}")

    # Save vocabulary and merges
    print(f"\n💾 Saving to disk...")

    # Save as pickle (for easy loading in Python)
    vocab_path = output_dir / f"{input_basename}_vocab.pkl"
    merges_path = output_dir / f"{input_basename}_merges.pkl"

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"   ✓ Saved vocabulary to {vocab_path}")

    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)
    print(f"   ✓ Saved merges to {merges_path}")

    # Save as JSON (for inspection)
    vocab_json_path = output_dir / f"{input_basename}_vocab.json"
    merges_json_path = output_dir / f"{input_basename}_merges.json"

    # Convert vocab to JSON-serializable format
    vocab_json = {
        str(k): v.decode('utf-8', errors='replace')
        for k, v in vocab.items()
    }
    with open(vocab_json_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved vocabulary (JSON) to {vocab_json_path}")

    # Convert merges to JSON-serializable format
    merges_json = [
        [a.decode('utf-8', errors='replace'), b.decode('utf-8', errors='replace')]
        for a, b in merges
    ]
    with open(merges_json_path, 'w', encoding='utf-8') as f:
        json.dump(merges_json, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved merges (JSON) to {merges_json_path}")

    # Save summary
    summary = {
        "training_time_seconds": training_time,
        "training_time_hours": training_time / 3600,
        "peak_memory_mb": peak_memory_mb,
        "end_memory_mb": end_memory_mb,
        "peak-end_memory_mb": peak_memory_mb - end_memory_mb,
        "process_rss_initial_mb": initial_memory,
        "process_rss_final_mb": final_memory_psutil,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "longest_token_id": longest_token_id,
        "longest_token_length": len(longest_token),
        "longest_token_text": longest_token.decode('utf-8', errors='replace'),
        "input_file": str(input_path),
        "input_file_size_mb": file_size_mb,
        "special_tokens": special_tokens,
    }

    summary_path = output_dir / f"{input_basename}_training_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved summary to {summary_path}")

    print(f"\n" + "=" * 80)
    print(f"✅ Training complete! Files saved to {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     run_train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 10000, ["<|endoftext|>"])