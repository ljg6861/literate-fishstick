#!/usr/bin/env python3
"""Simple profiling test."""
import os
os.environ['WANDB_MODE'] = 'disabled'
os.environ['DISABLE_COMPILE'] = '1'

import sys
sys.path.insert(0, '/home/lucas/literate-fishstick/TinyRecursiveModels')
os.chdir('/home/lucas/literate-fishstick/TinyRecursiveModels')

import torch
import json
import time
import gc

print('=== GPU INFO ===', flush=True)
print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB', flush=True)

print('\n=== BATCH SIZE MEMORY TEST ===', flush=True)

from utils.functions import load_model_class

with open('data/arc1concept-aug-1000/train/dataset.json', 'r') as f:
    meta = json.load(f)

for batch_size in [32, 64, 128, 192]:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        model_cfg = dict(
            batch_size=batch_size,
            vocab_size=meta['vocab_size'],
            seq_len=meta['seq_len'],
            num_puzzle_identifiers=meta['num_puzzle_identifiers'],
            causal=False,
            H_cycles=3, L_cycles=6, H_layers=0, L_layers=2,
            hidden_size=512, num_heads=8, expansion=4,
            puzzle_emb_ndim=512, puzzle_emb_len=16,
            pos_encodings='rope', forward_dtype='bfloat16',
            mlp_t=False, no_ACT_continue=True,
            halt_max_steps=16, halt_exploration_prob=0.1,
        )
        
        with torch.device('cuda'):
            model_cls = load_model_class('recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1')
            loss_cls = load_model_class('losses@ACTLossHead')
            model = model_cls(model_cfg)
            model = loss_cls(model, loss_type='stablemax_cross_entropy')
            model.train()
        
        batch = {
            'inputs': torch.randint(0, 12, (batch_size, 900), device='cuda'),
            'labels': torch.randint(0, 12, (batch_size, 900), device='cuda'),
            'puzzle_identifiers': torch.randint(1, 1000, (batch_size,), device='cuda'),
        }
        
        with torch.device('cuda'):
            carry = model.model.initial_carry(batch)
        
        # Forward + backward
        torch.cuda.synchronize()
        t0 = time.time()
        carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
        torch.cuda.synchronize()
        t1 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.time()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        forward_ms = (t1 - t0) * 1000
        backward_ms = (t2 - t1) * 1000
        
        # Get gradient dtype
        grad_dtype = None
        for p in model.parameters():
            if p.grad is not None:
                grad_dtype = p.grad.dtype
                break
        
        print(f'batch={batch_size}: fwd={forward_ms:.0f}ms, bwd={backward_ms:.0f}ms, mem={peak_mem:.1f}GB, grad={grad_dtype}', flush=True)
        
        del model, batch, carry
        
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'batch={batch_size}: OUT OF MEMORY', flush=True)
        else:
            print(f'batch={batch_size}: ERROR - {str(e)[:100]}', flush=True)
            import traceback
            traceback.print_exc()
    
    torch.cuda.empty_cache()
    gc.collect()

print('\nDone!', flush=True)
