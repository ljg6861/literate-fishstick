"""
ThinkSort v11: Simultaneous Multi-Task Training

Key Changes from v10:
- Train on ALL lengths (4, 8, 16, 32) in EVERY training step
- Each batch contains samples from all lengths
- Prevents catastrophic forgetting

Architecture: Same as v10 (Pre-LN Transformer with RoPE + Pointer)
Training: 25% N=4, 25% N=8, 25% N=16, 25% N=32 per batch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from dataclasses import dataclass
from collections import deque
import time


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build(max_len)
    
    def _build(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos', emb.cos()[None, None], persistent=False)
        self.register_buffer('sin', emb.sin()[None, None], persistent=False)
        self.max_len = seq_len
    
    def forward(self, q, k):
        L = q.size(2)
        if L > self.max_len:
            self._build(L)
        cos, sin = self.cos[:, :, :L], self.sin[:, :, :L]
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.rope = RoPE(self.head_dim)
    
    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q, k)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class Block(nn.Module):
    def __init__(self, dim, heads, ff):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Pre-LN
        x = x + self.ff(self.ln2(x))
        return x


@dataclass
class Config:
    dim: int = 128
    heads: int = 8
    ff: int = 512
    layers: int = 4
    vocab: int = 10
    lengths: tuple = (4, 8, 16, 32)
    samples_per_length: int = 128  # Per length per batch
    lr: float = 1e-4
    device: str = "cuda"


class PointerNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg.dim, cfg.heads, cfg.ff) for _ in range(cfg.layers)])
        self.q_proj = nn.Linear(cfg.dim, cfg.dim)
        self.k_proj = nn.Linear(cfg.dim, cfg.dim)
        self.sel_emb = nn.Parameter(torch.zeros(cfg.dim))
        self._init()
    
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def params(self):
        return sum(p.numel() for p in self.parameters())
    
    def encode(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return h
    
    def pointer(self, h, sel):
        B, L, D = h.shape
        mod = h + sel.unsqueeze(-1) * self.sel_emb
        rem = 1 - sel
        ctx = (mod * rem.unsqueeze(-1)).sum(1) / rem.sum(1, keepdim=True).clamp(min=1)
        q = self.q_proj(ctx)
        k = self.k_proj(mod)
        logits = (q.unsqueeze(1) @ k.transpose(1, 2)).squeeze(1) / math.sqrt(D)
        return logits.masked_fill(sel.bool(), -1e4)


class SimultaneousTrainer:
    """Train on ALL lengths simultaneously to prevent forgetting."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.device = self.cfg.device
        self.model = PointerNet(self.cfg).to(self.device)
        self.opt = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=0.01)
        
        print(f"🎯 PointerNet v11: Simultaneous Multi-Task")
        print(f"   Training on: {self.cfg.lengths} (all together!)")
        print(f"   Parameters: {self.model.params():,}")
        
        self.samples = 0
        self.acc_history = {n: deque(maxlen=50) for n in self.cfg.lengths}
    
    def train_step(self):
        """Single training step on ALL lengths simultaneously."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = {n: 0 for n in self.cfg.lengths}
        total_count = {n: 0 for n in self.cfg.lengths}
        
        # Train on each length
        for length in self.cfg.lengths:
            bs = self.cfg.samples_per_length
            
            # Generate batch
            x = torch.randint(0, self.cfg.vocab, (bs, length), device=self.device)
            target = torch.argsort(x, dim=1)
            
            # Encode
            h = self.model.encode(x)
            
            # Pointer decoding
            loss = 0.0
            correct = 0
            sel = torch.zeros(bs, length, device=self.device)
            
            for t in range(length):
                logits = self.model.pointer(h, sel)
                loss = loss + F.cross_entropy(logits, target[:, t])
                
                preds = logits.argmax(-1)
                correct += (preds == target[:, t]).sum().item()
                
                # Teacher forcing
                sel = sel.clone()
                sel[torch.arange(bs, device=self.device), target[:, t]] = 1
            
            total_loss = total_loss + loss / length
            total_correct[length] = correct
            total_count[length] = bs * length
        
        # Backprop on combined loss
        avg_loss = total_loss / len(self.cfg.lengths)
        
        self.opt.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        
        # Track accuracies
        for n in self.cfg.lengths:
            acc = total_correct[n] / total_count[n]
            self.acc_history[n].append(acc)
            self.samples += total_count[n]
        
        return avg_loss.item(), {n: total_correct[n] / total_count[n] for n in self.cfg.lengths}
    
    @torch.no_grad()
    def evaluate(self, length, n=2000):
        self.model.eval()
        total_correct = 0
        bs = min(500, n)
        
        for _ in range(n // bs):
            x = torch.randint(0, self.cfg.vocab, (bs, length), device=self.device)
            target = torch.argsort(x, dim=1)
            h = self.model.encode(x)
            sel = torch.zeros(bs, length, device=self.device)
            
            for t in range(length):
                logits = self.model.pointer(h, sel)
                preds = logits.argmax(-1)
                total_correct += (preds == target[:, t]).sum().item()
                sel[torch.arange(bs, device=self.device), preds.clamp(0, length-1)] = 1
        
        return total_correct / (n * length)
    
    def train(self, max_steps=50000, log_every=1000):
        print(f"\n{'='*65}")
        print("Simultaneous Multi-Task Training")
        print(f"Training on ALL lengths in EVERY step!")
        print(f"{'='*65}\n")
        
        start = time.time()
        losses = deque(maxlen=100)
        step = 0
        
        while step < max_steps:
            loss, accs = self.train_step()
            losses.append(loss)
            step += 1
            
            if step % log_every == 0:
                elapsed = time.time() - start
                speed = self.samples / elapsed
                
                # Evaluate all lengths
                eval_accs = {n: self.evaluate(n, 1000) for n in self.cfg.lengths}
                
                acc_str = " | ".join([f"N{n}:{eval_accs[n]:.0%}" for n in self.cfg.lengths])
                print(f"Step {step:5} | {acc_str} | Loss: {np.mean(losses):.4f} | {speed:,.0f}/s")
                
                # Check if we hit 90% on all
                if all(eval_accs[n] >= 0.90 for n in self.cfg.lengths):
                    print(f"\n🌟 ACHIEVED 90%+ ON ALL LENGTHS!")
                    break
        
        elapsed = time.time() - start
        print(f"\n{'='*65}")
        print(f"Training Complete! Time: {elapsed:.1f}s, Steps: {step}")
        print(f"{'='*65}")
    
    def full_eval(self):
        print("\n" + "="*65)
        print("FINAL EVALUATION")
        print("="*65)
        
        print("\n📊 Trained lengths:")
        for n in self.cfg.lengths:
            acc = self.evaluate(n, 2000)
            status = '✅' if acc >= 0.90 else ('🔶' if acc >= 0.80 else '❌')
            print(f"  N={n:2}: {acc:.2%} {status}")
        
        print("\n🌟 Zero-shot OOD:")
        for n in [48, 64, 96, 128]:
            acc = self.evaluate(n, 500)
            status = '✅' if acc >= 0.70 else ('🔶' if acc >= 0.50 else '❌')
            print(f"  N={n:3}: {acc:.2%} {status}")


def run():
    cfg = Config(
        dim=128, heads=8, ff=512, layers=4,
        vocab=10, lengths=(4, 8, 16, 32),
        samples_per_length=128,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    t = SimultaneousTrainer(cfg)
    t.train(max_steps=100000, log_every=2000)
    t.full_eval()
    return t


if __name__ == "__main__":
    run()
