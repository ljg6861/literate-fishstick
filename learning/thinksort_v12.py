"""
ThinkSort v12: Gradient-Balanced Simultaneous Training

Key Innovation: Adaptive Loss Weighting
- Higher weights for smaller lengths initially (lock in fundamentals)
- Gradually increase N=32 weight to push for perfection
- Prevents N=32 gradients from "bullying" smaller tasks

Formula: L_total = w1*L_N4 + w2*L_N8 + w3*L_N16 + w4*L_N32

Weight Schedule:
- Phase 1 (0-10k): Focus on fundamentals (w=[0.4, 0.3, 0.2, 0.1])
- Phase 2 (10k-30k): Balance (w=[0.25, 0.25, 0.25, 0.25])
- Phase 3 (30k+): Push N=32 (w=[0.15, 0.15, 0.25, 0.45])
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
        x = x + self.attn(self.ln1(x))
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
    samples_per_length: int = 128
    lr: float = 1e-4
    device: str = "cuda"
    
    # Weight schedule phases
    phase1_steps: int = 10000
    phase2_steps: int = 30000
    # Weights: [N4, N8, N16, N32]
    weights_phase1: tuple = (0.4, 0.3, 0.2, 0.1)   # Lock in fundamentals
    weights_phase2: tuple = (0.25, 0.25, 0.25, 0.25)  # Balance
    weights_phase3: tuple = (0.10, 0.10, 0.25, 0.55)  # Push N32


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


class GradientBalancedTrainer:
    """Gradient-balanced multi-task training with adaptive loss weighting."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.device = self.cfg.device
        self.model = PointerNet(self.cfg).to(self.device)
        self.opt = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=0.01)
        
        print(f"🎯 PointerNet v12: Gradient-Balanced Training")
        print(f"   Adaptive Loss Weighting: ENABLED")
        print(f"   Parameters: {self.model.params():,}")
        
        self.step = 0
        self.samples = 0
        self.acc_history = {n: deque(maxlen=50) for n in self.cfg.lengths}
    
    def get_weights(self):
        """Get current loss weights based on training phase."""
        if self.step < self.cfg.phase1_steps:
            return self.cfg.weights_phase1
        elif self.step < self.cfg.phase2_steps:
            return self.cfg.weights_phase2
        else:
            return self.cfg.weights_phase3
    
    def train_step(self):
        """Single training step with weighted losses."""
        self.model.train()
        
        weights = self.get_weights()
        total_loss = 0.0
        total_correct = {n: 0 for n in self.cfg.lengths}
        total_count = {n: 0 for n in self.cfg.lengths}
        
        for i, length in enumerate(self.cfg.lengths):
            bs = self.cfg.samples_per_length
            x = torch.randint(0, self.cfg.vocab, (bs, length), device=self.device)
            target = torch.argsort(x, dim=1)
            
            h = self.model.encode(x)
            loss = 0.0
            correct = 0
            sel = torch.zeros(bs, length, device=self.device)
            
            for t in range(length):
                logits = self.model.pointer(h, sel)
                loss = loss + F.cross_entropy(logits, target[:, t])
                
                preds = logits.argmax(-1)
                correct += (preds == target[:, t]).sum().item()
                
                sel = sel.clone()
                sel[torch.arange(bs, device=self.device), target[:, t]] = 1
            
            # Apply weight to this length's loss
            weighted_loss = weights[i] * (loss / length)
            total_loss = total_loss + weighted_loss
            total_correct[length] = correct
            total_count[length] = bs * length
        
        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        
        self.step += 1
        for n in self.cfg.lengths:
            acc = total_correct[n] / total_count[n]
            self.acc_history[n].append(acc)
            self.samples += total_count[n]
        
        return total_loss.item(), {n: total_correct[n] / total_count[n] for n in self.cfg.lengths}
    
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
    
    def train(self, max_steps=100000, log_every=2000):
        print(f"\n{'='*70}")
        print("Gradient-Balanced Multi-Task Training")
        print(f"Weight Schedule:")
        print(f"  Phase 1 (0-{self.cfg.phase1_steps//1000}k): {self.cfg.weights_phase1}")
        print(f"  Phase 2 ({self.cfg.phase1_steps//1000}k-{self.cfg.phase2_steps//1000}k): {self.cfg.weights_phase2}")
        print(f"  Phase 3 ({self.cfg.phase2_steps//1000}k+): {self.cfg.weights_phase3}")
        print(f"{'='*70}\n")
        
        start = time.time()
        losses = deque(maxlen=100)
        
        while self.step < max_steps:
            loss, accs = self.train_step()
            losses.append(loss)
            
            if self.step % log_every == 0:
                elapsed = time.time() - start
                speed = self.samples / elapsed
                
                eval_accs = {n: self.evaluate(n, 1000) for n in self.cfg.lengths}
                weights = self.get_weights()
                
                phase = "P1" if self.step < self.cfg.phase1_steps else ("P2" if self.step < self.cfg.phase2_steps else "P3")
                acc_str = " | ".join([f"N{n}:{eval_accs[n]:.0%}" for n in self.cfg.lengths])
                
                print(f"Step {self.step:5} [{phase}] | {acc_str} | Loss: {np.mean(losses):.4f} | {speed:,.0f}/s")
                
                if all(eval_accs[n] >= 0.90 for n in self.cfg.lengths):
                    print(f"\n🌟 ACHIEVED 90%+ ON ALL LENGTHS!")
                    break
        
        elapsed = time.time() - start
        print(f"\n{'='*70}")
        print(f"Training Complete! Time: {elapsed:.1f}s")
        print(f"{'='*70}")
    
    def full_eval(self):
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        
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
    t = GradientBalancedTrainer(cfg)
    t.train(max_steps=100000, log_every=2000)
    t.full_eval()
    return t


if __name__ == "__main__":
    run()
