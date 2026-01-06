#!/usr/bin/env python3
import torch
from model import Config, UniversalPointerNet
from trainer import Trainer
import sys
import os

def test_checkpoint(path="checkpoint_latest.pt"):
    if not os.path.exists(path):
        print(f"❌ Checkpoint not found: {path}")
        return

    print(f"🔎 Loading checkpoint: {path}")
    
    # Load checkpoint just to get config if possible, or assume default/infer
    # references trainer.load logic
    checkpoint = torch.load(path, map_location='cpu')
    config = checkpoint.get('config')
    
    if config is None:
        print("⚠️ No config in checkpoint, using default...")
        config = Config()
        
    # Ensure continuous (no vocab) if old config
    if hasattr(config, 'vocab'):
         del config.vocab

    model = UniversalPointerNet(config)
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model'])
    except Exception as e:
        print(f"⚠️ Strict load failed, trying loose: {e}")
        model.load_state_dict(checkpoint['model'], strict=False)
        
    model.to(config.device)
    model.eval()
    
    print(f"✅ Model Loaded! (Dim={config.dim}, Heads={config.heads}, Rec={config.recurrent_steps})")
    
    trainer = Trainer(config)
    trainer.model = model # Inject loaded model
    # If using DataParallel in trainer, we might need to wrap it, but evaluate() handles it or uses raw_model?
    # Trainer init wraps model. We just want to use the trainer's eval methods.
    # Note: Trainer creates its own model in __init__. We should overwrite it.
    if torch.cuda.device_count() > 1:
        import torch.nn as nn
        trainer.model = nn.DataParallel(model)
        trainer.raw_model = model
    else:
        trainer.model = model
        trainer.raw_model = model
        
    print("\n📊 Running Zero-Shot Evaluation:")
    trainer.zero_shot_eval()
    
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint_latest.pt"
    test_checkpoint(path)
