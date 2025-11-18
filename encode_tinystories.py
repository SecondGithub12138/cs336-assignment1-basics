from tests.adapters import Tokenizer
import numpy as np
import os


vocab_path = "outputs/TinyStoriesV2-GPT4-train/TinyStoriesV2-GPT4-train_vocab.pkl"
merges = "outputs/TinyStoriesV2-GPT4-train/TinyStoriesV2-GPT4-train_merges.pkl"

encode_file_path = "data/TinyStoriesV2-GPT4-valid.txt"
output_text_path = "encode/TinyStoriesV2-GPT4-train_merges.txt"
output_bin_path = "encode/TinyStoriesV2-GPT4-train_merges.bin"

if __name__ == "__main__":
    os.makedirs("encode", exist_ok=True)
    tokenizer = Tokenizer.from_files(vocab_path, merges, ["<|endoftext|>"])

    batch_size = 10000  # 每次写入1万个token

    # 方式1: 写成文本格式（human readable，方便检查）
    print("Writing text format...")
    with open(output_text_path, "w") as f:
        batch = []
        for token_id in tokenizer.encode_large_file(encode_file_path, 4, b"<|endoftext|>"):
            batch.append(token_id)
            if len(batch) >= batch_size:
                # 批量写入（会追加到文件末尾）
                f.write("\n".join(map(str, batch)) + "\n")
                batch = []
        # 写入剩余的
        if batch:
            f.write("\n".join(map(str, batch)) + "\n")

    # 方式2: 写成二进制格式（推荐，省空间）
    # 为什么 uint16 合适：
    # - vocab_size <= 32K，在 uint16 范围内 (0-65535)
    # - 每个 token 只占 2 bytes
    print("Writing binary format (uint16)...")
    with open(output_bin_path, "wb") as f:
        batch = []
        for token_id in tokenizer.encode_large_file(encode_file_path, 4, b"<|endoftext|>"):
            batch.append(token_id)
            if len(batch) >= batch_size:
                # 批量写入（会追加到文件末尾）
                np.array(batch, dtype=np.uint16).tofile(f)
                batch = []
        # 写入剩余的
        if batch:
            np.array(batch, dtype=np.uint16).tofile(f)

    print("Done!")
