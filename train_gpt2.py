import torch 
torch.cuda.empty_cache()
import torch.nn as nn
import tiktoken
from torch.nn import functional as F
from dataclasses import dataclass
from hellaswag import render_example, iterate_example
import inspect
import math
import time
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f'config.n_embd % config.n_head != 0'
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
    def forward(self, x):
        # x -> (B, T, embd)
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, 2) # q, k, v -> (B, T, embd)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y) # (B, T, n_embd)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        # x -> (B, T, n_embd)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        # x -> (B, T, n_embd)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_head: int = 12
    n_embd: int = 768
    n_layer: int = 12

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f  = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** (-0.5)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, target=None):
        # idx -> (B, T), target -> (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, f"T > self.config.block_size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = pos_emb + tok_emb # (B, T , n_embd)
        for b in self.transformer.h:
            x = b(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if target is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, module_type):
        """load parameters from huggingface"""
        assert module_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[module_type]
        config_args['block_size'] = 1024
        config_args['vocab_size'] = 50257
        config = GPTConfig(**config_args)
        model = GPT(config)
        st = model.state_dict()
        st_keys = st.keys()
        st_keys = [s for s in st_keys if not s.endswith(".attn.bias")]
        from transformers import GPT2LMHeadModel
        print(f"loading params from huggingface {module_type}")
        model_hf = GPT2LMHeadModel.from_pretrained(module_type)
        st_hf = model_hf.state_dict()
        st_keys_hf = st_hf.keys()
        st_keys_hf = [s for s in st_keys_hf if not s.endswith(".attn.masked_bias")]
        st_keys_hf = [s for s in st_keys_hf if not s.endswith(".attn.bias")]
        transpose_list = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_proj.weight", "mlp.c_fc.weight"]
        assert len(st_keys) == len(st_keys_hf), f"len(st_keys) != len(st_keys_hf)"
        for k in st_keys:
            if any(k.endswith(t) for t in transpose_list):
                assert st[k].shape[::-1] == st_hf[k].shape
                with torch.no_grad():
                    st[k].copy_(st_hf[k].t())
            else:
                assert st[k].shape == st_hf[k].shape
                with torch.no_grad():
                    st[k].copy_(st_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        p_dict = {pn: p for pn, p in self.named_parameters()}
        p_dict = {pn: p for pn, p in p_dict.items() if p.requires_grad == True}
        decay_params = [p for _, p in p_dict.items() if p.dim() >= 2 ]
        nondecay_params = [p for _, p in p_dict.items() if p.dim() < 2]
        optim_groups = [
            dict(params=decay_params, weight_decay=weight_decay),
            dict(params=nondecay_params, weight_decay=0.0)]
        num_decay_params = sum([p.numel() for p in decay_params])
        num_nondecay_params = sum([p.numel() for p in nondecay_params])
        if master_process:
            print(f"decay_params: {decay_params}, num_decay_params: {num_decay_params}")
            print(f"nondecay_params: {nondecay_params}, num_nondecay_params: {num_nondecay_params}")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        if master_process:
            print(f"use_fused: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
# ------------------------------------------------------------------------------------------------------------------
import numpy as np
import os
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int64)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    def __init__(self, B, T, rank_process, num_process, split):
        assert split in {'val', 'train'}
        self.B = B
        self.T = T
        self.rank_process = rank_process
        self.num_process = num_process
        self.split = split
        
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(self.shards) > 0, f"len(self.shards) == 0"
        if len(self.shards) > 0:
            if master_process:
                print(f"the length of shards is {len(shards) } in {split} split") 
        self.reset()
    def reset(self):
        self.current_shard = 0
        self.current_position = self.B * self.T * self.rank_process
        self.tokens = load_tokens(self.shards[self.current_shard])
    def next_batch(self):
        B, T = self.B, self.T
        if self.current_position + B*T*self.num_process + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.current_position = B*T*self.rank_process
            self.tokens = load_tokens(self.shards[self.current_shard])
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B*T*self.num_process
        return x, y
# ---------------------------------------------------------------------------------------------------------
# help function for hellaswag eval
def get_most_likely_row(tokens, mask, logits):
    shift_tokens = (tokens[:, 1:]).contiguous() # B, T-1
    shitf_logits = (logits[:, :-1, :]).contiguous() # B, T-1, vocab_size
    shift_mask = (mask[:, 1:]).contiguous() # B, T-1
    flat_shift_tokens = shift_tokens.view(-1) # B*T-1
    flat_shift_logits = shitf_logits.view(-1, shitf_logits.shape[-1]) # B*T-1, vocab_size
    loss = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none") # B*T-1
    loss = loss.view(logits.shape[0], -1) # B, T-1
    mask_loss = loss * shift_mask
    mask_loss_sum = mask_loss.sum(-1) # B, 1
    mask_loss_avg = mask_loss_sum / shift_mask.sum(-1)
    pred_norm = mask_loss_avg.argmin().item()
    pred = mask_loss_sum.argmin().item()
    return pred_norm
# ---------------------------------------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), f"we need cuda for ddp"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device {device}")
device_type = "cuda" if device.startswith("cuda") else "cpu"


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
enc =  tiktoken.get_encoding("gpt2")
total_batch_size = 65536 
B = 4
T = 512
assert total_batch_size % (B * T * ddp_world_size) == 0, f"total_batch_size should be divided by B*T*ddp_world_size"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
print(f"the total batch size is {total_batch_size}")
print(f"grad accumlation steps is {grad_accum_steps}")

train_loader = DataLoader(B, T, ddp_rank, ddp_world_size, 'train')
val_loader = DataLoader(B, T, ddp_rank, ddp_world_size, 'val')

torch.set_float32_matmul_precision("high")

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
use_compile = False
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(module=model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 225 # 改
max_steps = 6103 # 改

def get_lr(it):
    if it < warmup_steps:
        lr = max_lr * (it+1) / warmup_steps
    elif it > max_steps:
        lr = min_lr
    else:
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = min_lr + coeff * (max_lr - min_lr)
    return lr

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = step == max_steps-1

    if step%80 == 0 or last_step:
        model.eval()
        loss_accum = torch.tensor(0.0, device=device)
        loss_step = 20
        val_loader.reset()  
        with torch.no_grad():
            for i in range(loss_step):
                x, y = val_loader.next_batch()
                x = x.to(device)
                y = y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss /= loss_step
                loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"step {step} :  val loss is {loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"step {step}: val loss is {loss_accum.item():.4f}\n")
            if step > 0 and (step % 1600 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": loss_accum.item(),
                }
                torch.save(checkpoint, checkpoint_path)
    # hellaswag 
    if (step%80 == 0 or last_step) and not use_compile:
        model.eval()
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_example('val')):
            if i % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = render_example(example)
            mask = mask.to(device)
            tokens = tokens.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
            acc_norm = num_correct_norm / num_total

        if ddp:
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            num_correct_norm = num_correct_norm.item()
            num_total = num_total.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"step {step}: hellaswag acc: {num_correct_norm} / {num_total} : {acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"step {step}: hellaswag acc: {num_correct_norm} / {num_total} : {acc_norm:.4f}")
    # generate from the model
    if step > 0 and(step%80 == 0 or last_step) and not use_compile:
        model.eval()
        return_sentences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,") 
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        xgen = tokens.unsqueeze(0).repeat(return_sentences, 1)
        xgen = xgen.to(device)
        g = torch.Generator(device=device)
        g.manual_seed(42 + ddp_rank)
        while xgen.shape[1] < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # B, T, vocab_size
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1) # B, vocab_size
                topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1) # B, 50
                ix = torch.multinomial(topk_probs, num_samples=1, generator=g) # B, 1
                idx = torch.gather(topk_indices, dim=-1, index=ix) # B,1
                xgen = torch.cat((xgen, idx), dim=1)
        xgen_cpu = xgen.cpu()
        for i in range(return_sentences):
            ids = xgen_cpu[i, :max_length].tolist()
            text = enc.decode(ids)
            print(f"rank {ddp_rank} sample {i} : {text}")

    model.train()
    optimizer.zero_grad()
    loss_accum = torch.tensor(0.0, device=device)
    for i in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss /= grad_accum_steps
        loss_accum += loss.detach()
        # if ddp:
        #     model.require_backward_grad_sync = i == grad_accum_steps-1
        if ddp and i != grad_accum_steps-1:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for param_group in optimizer.param_groups:
        param_group["lr"] = get_lr(step)
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step: {step} | train loss: {loss_accum.item()} | norm: {norm.item()} | dt: {dt*1000}ms | tokens_per_sec: {tokens_per_sec}")
        with open(log_file, "a") as f:
            f.write(f"step {step}: train loss {loss_accum.item()}\n")
if ddp:
    destroy_process_group()
