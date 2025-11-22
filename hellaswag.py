"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""
import torch
import tiktoken
import requests
import os
import json
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from torch.nn import functional as F

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag") 
def download_file(url, fname, chunk_size=1024):
    requ = requests.get(url, stream=True)
    with open(fname, "wb") as f, tqdm(
        desc=fname,
        total=int(requ.headers.get("content-length", 0)),
        unit="iB",
        unit_scale=True,
        unit_divisor=1024) as bar:
        for data in requ.iter_content(chunk_size):
            size = f.write(data)
            bar.update(size)
            
hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    url = hellaswags[split]
    filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(filename):
        print(f"downloading {split} jsonl file .....")
        download_file(url, filename)

enc = tiktoken.get_encoding("gpt2")
def render_example(example):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]
    data = {
        "label":label,
        "ctx_tokens":None,
        "ending_tokens":[]
    }
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    mask_rows = []
    tok_rows = []
    for end in endings:
        end_row = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_row)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_row))
        data["ending_tokens"].append(end_row)
    # 长度统一
    maxlength = max(len(r) for r in tok_rows)
    tokens = torch.zeros((4, maxlength), dtype=torch.long)
    mask = torch.zeros((4, maxlength), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
    return data, tokens, mask, label

def iterate_example(split):
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate(model_type, device):
    torch.set_float32_matmul_precision("high")
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model = model.to(device)
    num_correct = 0
    num_correct_norm = 0
    num_total = 0
    for example in iterate_example("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        logits = model(tokens).logits
        # logits -> B, T, V
        shift_logits = (logits[:, :-1, :]).contiguous() # B, T-1, V
        shift_tokens = (tokens[:, 1:]).contiguous() # B, T-1
        flat_shift_logits = shift_logits.view(-1, shift_logits.shape[-1]) # B*T-1, V
        flat_shift_tokens = shift_tokens.view(-1) # B*T-1
        loss = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none") # B*T-1
        loss = loss.view(logits.shape[0], -1) # B, T-1
        shift_mask = mask[:, 1:].contiguous() # B, T-1
        mask_shift_loss = loss * shift_mask # B, T-1
        sum_loss = mask_shift_loss.sum(-1) # B, 1
        avg_loss = sum_loss / shift_mask.sum(-1) # B, 1
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        if num_total % 1000 == 0:
            print(f"{num_total}: acc: {num_correct_norm} / {num_total} : {num_correct_norm/num_total:.4f} ")

        if num_total < 10:
            print("----------------------")
            print("context", example["ctx"])
            print("Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i}: {end}")
                print(f"avg_loss: {avg_loss[i].item()}")
            print(f"pred: {pred_norm} label: {label}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)
        

        







