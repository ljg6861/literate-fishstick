# ThinkSort: Universal Transformer for Neural Sorting

A neural network that learns to sort sequences using a **Universal Transformer** architecture with:
- **RoPE (Rotary Position Embeddings)** for length-invariant positions
- **Weight-shared recurrent blocks** for algorithmic generalization
- **Pointer Network** with hard masking for selection-sort-like behavior

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training and evaluation
python run.py
```

## Results

Training on N=4,8,16 for 50,000 steps:

| Length | Accuracy | Notes |
|--------|----------|-------|
| N=4 | 95% | Trained |
| N=8 | 88% | Trained |
| N=16 | 70% | Trained |
| N=32 | 27% | Zero-shot |
| N=100 | 6% | Zero-shot |
| N=1000 | 0.4% | Zero-shot |

## Architecture

```
Input Numbers → Token Embedding → Universal Block (×4 recurrent) → Pointer → Output Position
                     ↑                    ↓
                     └── RoPE + Hard Mask ─┘
```

### Key Components

1. **Universal Transformer Block**: Single weight-shared layer run N times
2. **RoPE**: Rotary position embeddings for relative position encoding
3. **Hard Masking**: Already-selected positions masked with -∞
4. **Pointer Network**: Outputs attention over remaining positions

## Files

- `model.py` - Universal Transformer Pointer Network
- `trainer.py` - Training and evaluation logic
- `run.py` - Entry point to reproduce results
- `requirements.txt` - Dependencies

## Configuration

Edit `run.py` to customize:

```python
config = Config(
    dim=128,           # Model dimension
    heads=8,           # Attention heads
    ff=512,            # FFN dimension
    recurrent_steps=4, # Universal block iterations
    vocab=10,          # Number vocabulary (0-9)
    train_lengths=(4, 8, 16),  # Training sequence lengths
)
```

## License

MIT
