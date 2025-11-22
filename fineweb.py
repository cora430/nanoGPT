import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import tiktoken
import numpy as np
import multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
shard_size = int(1e7)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

def tokenize(doc):
    tokens = [eot]
    text = doc["text"]
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens, dtype=np.int64)
    if not (tokens_np.min() >= 0 and tokens_np.max() < 2**16):
        raise ValueError("token dictionary too large for uint16")
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    np.save(os.path.join(DATA_CACHE_DIR, filename), tokens_np)

def process_dataset(fw_iterable):
    nproces = max(1, os.cpu_count() // 2)
    shard_index = 0
    count_tokens = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    progress_bar = None

    with mp.Pool(nproces) as pool:
        for tokens in pool.imap(tokenize, fw_iterable, chunksize=16):
            tlen = len(tokens)
            if tlen + count_tokens <= shard_size:
                all_tokens_np[count_tokens:count_tokens + tlen] = tokens
                count_tokens += tlen
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"shard {shard_index}")
                progress_bar.update(tlen)
            else:
                reminder = shard_size - count_tokens
                if reminder > 0:
                    all_tokens_np[count_tokens:] = tokens[:reminder]
                split = 'val' if shard_index == 0 else 'train'
                filename = f"edufineweb_{split}_{shard_index:06d}.npy"
                write_datafile(filename, all_tokens_np)
                if progress_bar is not None:
                    progress_bar.update(reminder)
                    progress_bar.close()

                remaining = tlen - reminder
                all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
                if remaining > 0:
                    all_tokens_np[:remaining] = tokens[reminder:]
                count_tokens = remaining
                shard_index += 1
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"shard {shard_index}")
                progress_bar.update(count_tokens)

    if count_tokens != 0:
        split = 'val' if shard_index == 0 else 'train'
        filename = f"edufineweb_{split}_{shard_index:06d}.npy"
        write_datafile(filename, all_tokens_np[:count_tokens])

if __name__ == "__main__":
    # 不使用 split slice，加载整个 train（如果很大请注意内存/网络）
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    size = int(len(fw) * 0.01)   # 前 1%
    fw_small = fw.select(range(size))   # 安全取前 size 条
    process_dataset(fw_small)
