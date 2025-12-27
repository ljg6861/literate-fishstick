"""
ThinkSort v9: Simultaneous Multi-Length Training

Key insight: Curriculum is getting stuck. Instead:
- Train on ALL lengths (4,8,16,32) simultaneously in every batch
- Each sample is randomly assigned a length
- Model learns general sorting, not length-specific patterns

This is the "true" multi-task learning approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from typing import Tuple
from dataclasses import dataclass
from collections import deque
import time


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=512, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_len)
    
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin', emb.sin()[None, None, :, :], persistent=False)
        self.max_len = seq_len
    
    def forward(self, seq_len, device):
        if seq_len > self.max_len:
            self._build_cache(seq_len)
        return self.cos[:, :, :seq_len].to(device), self.sin[:, :, :seq_len].to(device)


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.rope = RotaryEmbedding(self.head_dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rope(L, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), 
            nn.Linear(ff_dim, dim), nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x


@dataclass 
class Config:
    dim: int = 128
    heads: int = 8
    ff_dim: int = 512
    layers: int = 4
    vocab: int = 10
    lengths: Tuple[int, ...] = (4, 8, 16, 32)
    batch: int = 256
    lr: float = 3e-4
    device: str = "cuda"


class SortNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab + 1, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg.dim, cfg.heads, cfg.ff_dim) for _ in range(cfg.layers)])
        self.ln = nn.LayerNorm(cfg.dim)
        self.q_proj = nn.Linear(cfg.dim, cfg.dim)
        self.k_proj = nn.Linear(cfg.dim, cfg.dim)
        self.selected_bias = nn.Parameter(torch.zeros(cfg.dim))
        self._init()
    
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def params(self):
        return sum(p.numel() for p in self.parameters())
    
    def encode(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.ln(h)
    
    def pointer(self, enc, sel_mask):
        B, L, D = enc.shape
        mod = enc + sel_mask.unsqueeze(-1) * self.selected_bias
        rem = 1 - sel_mask
        ctx = (mod * rem.unsqueeze(-1)).sum(1) / rem.sum(1, keepdim=True).clamp(min=1)
        q = self.q_proj(ctx)
        k = self.k_proj(mod)
        logits = (q.unsqueeze(1) @ k.transpose(1, 2)).squeeze(1) / math.sqrt(D)
        return logits.masked_fill(sel_mask.bool(), -1e9)


class Trainer:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.device = self.cfg.device
        self.model = SortNet(self.cfg).to(self.device)
        self.opt = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=0.01)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100000)
        self.samples = 0
        self.acc_history = {n: deque(maxlen=50) for n in self.cfg.lengths}
        
        print(f"🎯 SortNet v9: Simultaneous Multi-Length")
        print(f"   Lengths: {self.cfg.lengths} (all trained together)")
        print(f"   Parameters: {self.model.params():,}")
    
    def batch(self, length):
        x = torch.randint(0, self.cfg.vocab, (self.cfg.batch, length), device=self.device)
        return x, torch.argsort(x, dim=1)
    
    def step(self, length):
        self.model.train()
        x, target = self.batch(length)
        enc = self.model.encode(x)
        
        loss = 0.0
        correct = 0
        sel = torch.zeros(self.cfg.batch, length, device=self.device)
        
        for t in range(length):
            logits = self.model.pointer(enc, sel)
            loss = loss + F.cross_entropy(logits, target[:, t], label_smoothing=0.1)
            preds = logits.argmax(-1)
            correct += (preds == target[:, t]).sum().item()
            sel = sel.clone()
            sel[torch.arange(self.cfg.batch, device=self.device), target[:, t]] = 1
        
        loss = loss / length
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.sched.step()
        
        acc = correct / (self.cfg.batch * length)
        self.acc_history[length].append(acc)
        return loss.item(), acc
    
    @torch.no_grad()
    def evaluate(self, length, n=1000):
        self.model.eval()
        x = torch.randint(0, self.cfg.vocab, (n, length), device=self.device)
        target = torch.argsort(x, dim=1)
        enc = self.model.encode(x)
        sel = torch.zeros(n, length, device=self.device)
        correct = 0
        for t in range(length):
            logits = self.model.pointer(enc, sel)
            preds = logits.argmax(-1)
            correct += (preds == target[:, t]).sum().item()
            sel[torch.arange(n, device=self.device), preds.clamp(0, length-1)] = 1
        return correct / (n * length)
    
    def train(self, max_samples=20000000, log_every=200000):
        print(f"\n{'='*70}")
        print("SortNet v9: Training on all lengths simultaneously")
        print(f"{'='*70}\n")
        
        start = time.time()
        losses = deque(maxlen=100)
        
        while self.samples < max_samples:
            # Random length each step
            length = np.random.choice(self.cfg.lengths)
            loss, acc = self.step(length)
            losses.append(loss)
            self.samples += self.cfg.batch * length
            
            if self.samples % log_every < self.cfg.batch * max(self.cfg.lengths):
                elapsed = time.time() - start
                speed = self.samples / elapsed
                
                # Evaluate all lengths
                accs = {n: self.evaluate(n, 500) for n in self.cfg.lengths}
                acc_str = " | ".join([f"N{n}:{accs[n]:.0%}" for n in self.cfg.lengths])
                
                print(f"Samples: {self.samples:10,} | {acc_str} | Loss: {np.mean(losses):.3f} | {speed:,.0f}/s")
                
                # Check if we hit 90% on all
                if all(accs[n] >= 0.90 for n in self.cfg.lengths):
                    print(f"\n🌟 ACHIEVED 90%+ ON ALL LENGTHS!")
                    break
        
        print(f"\n{'='*70}")
        print(f"Training Complete! Time: {time.time()-start:.1f}s")
        print(f"{'='*70}")
    
    def full_eval(self):
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        
        print("\n📊 TRAINED LENGTHS:")
        for n in self.cfg.lengths:
            acc = self.evaluate(n, 2000)
            status = '✅' if acc >= 0.90 else ('🔶' if acc >= 0.80 else '❌')
            print(f"  Sort {n:3}: {acc:.2%} {status}")
        
        print("\n🌟 ZERO-SHOT OOD:")
        for n in [48, 64, 96, 128, 256]:
            acc = self.evaluate(n, 500)
            status = '🌟' if acc >= 0.9 else ('✅' if acc >= 0.7 else ('🔶' if acc >= 0.5 else '❌'))
            print(f"  Sort {n:3}: {acc:.2%} {status}")


def run():
    cfg = Config(
        dim=128, heads=8, ff_dim=512, layers=4,
        vocab=10, lengths=(4, 8, 16, 32), batch=256,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    t = Trainer(cfg)
    t.train(max_samples=30000000, log_every=500000)
    t.full_eval()
    return t


if __name__ == "__main__":
    run()
