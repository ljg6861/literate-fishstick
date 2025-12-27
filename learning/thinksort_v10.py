"""
ThinkSort v10: Stable Single-Length Training

Back to basics:
1. Train on ONE length at a time until mastered
2. Simple, stable learning rate
3. No fancy schedulers - just AdamW with warmup
4. Verify 95%+ before moving to next length

Goal: Get verifiable 95%+ on each length, then generalize
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
        x = x + self.attn(self.ln1(x))  # Pre-LN for stability
        x = x + self.ff(self.ln2(x))
        return x


@dataclass
class Config:
    dim: int = 128
    heads: int = 8
    ff: int = 512
    layers: int = 4
    vocab: int = 10
    batch: int = 512
    lr: float = 1e-4  # Lower, stable LR
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
        # h: (B, L, D), sel: (B, L) binary mask
        B, L, D = h.shape
        mod = h + sel.unsqueeze(-1) * self.sel_emb
        rem = 1 - sel
        ctx = (mod * rem.unsqueeze(-1)).sum(1) / rem.sum(1, keepdim=True).clamp(min=1)
        q = self.q_proj(ctx)  # (B, D)
        k = self.k_proj(mod)  # (B, L, D)
        logits = (q.unsqueeze(1) @ k.transpose(1, 2)).squeeze(1) / math.sqrt(D)
        logits = logits.masked_fill(sel.bool(), -1e4)
        return logits


class Trainer:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.device = self.cfg.device
        self.model = PointerNet(self.cfg).to(self.device)
        self.opt = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=0.01)
        print(f"🎯 PointerNet v10: Stable Training")
        print(f"   Parameters: {self.model.params():,}")
    
    def batch(self, length):
        x = torch.randint(0, self.cfg.vocab, (self.cfg.batch, length), device=self.device)
        return x, torch.argsort(x, dim=1)
    
    def train_step(self, length):
        self.model.train()
        x, target = self.batch(length)
        h = self.model.encode(x)
        
        total_loss = 0.0
        correct = 0
        sel = torch.zeros(self.cfg.batch, length, device=self.device)
        
        for t in range(length):
            logits = self.model.pointer(h, sel)
            loss = F.cross_entropy(logits, target[:, t])
            total_loss = total_loss + loss
            
            preds = logits.argmax(-1)
            correct += (preds == target[:, t]).sum().item()
            
            # Teacher forcing
            sel = sel.clone()
            sel[torch.arange(self.cfg.batch, device=self.device), target[:, t]] = 1
        
        avg_loss = total_loss / length
        
        self.opt.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        
        return avg_loss.item(), correct / (self.cfg.batch * length)
    
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
    
    def train_length(self, length, target_acc=0.95, max_steps=50000):
        print(f"\n▶ Training on N={length} (target: {target_acc:.0%})")
        
        recent_acc = deque(maxlen=100)
        recent_loss = deque(maxlen=100)
        steps = 0
        
        while steps < max_steps:
            loss, acc = self.train_step(length)
            recent_acc.append(acc)
            recent_loss.append(loss)
            steps += 1
            
            if steps % 500 == 0:
                avg_acc = np.mean(recent_acc)
                avg_loss = np.mean(recent_loss)
                eval_acc = self.evaluate(length, 1000)
                print(f"  Step {steps:5} | Train: {avg_acc:.1%} | Eval: {eval_acc:.1%} | Loss: {avg_loss:.4f}")
                
                if eval_acc >= target_acc:
                    print(f"  ✅ Reached {eval_acc:.1%} on N={length}")
                    return True
        
        final = self.evaluate(length, 2000)
        print(f"  ⏹ Stopped at {final:.1%} after {max_steps} steps")
        return final >= target_acc
    
    def train_curriculum(self, lengths=(4, 8, 16, 32), target=0.95):
        print("="*60)
        print("Curriculum Training: Master each length before advancing")
        print("="*60)
        
        for length in lengths:
            success = self.train_length(length, target_acc=target, max_steps=30000)
            if not success:
                print(f"❌ Failed to reach {target:.0%} on N={length}")
                break
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        print("\n📊 Trained lengths:")
        for n in lengths:
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
        vocab=10, batch=512, lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    t = Trainer(cfg)
    t.train_curriculum(lengths=(4, 8, 16, 32), target=0.95)
    return t


if __name__ == "__main__":
    run()
