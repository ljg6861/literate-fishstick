"""
Supervised Learning Baseline for Seq2Seq Tasks

This module implements a standard Transformer with teacher forcing
for fair comparison against MuZero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import time

from .muzero_transformer import PositionalEncoding, TransformerBlock
from .seq2seq_envs import create_env


@dataclass
class SupervisedConfig:
    """Configuration for supervised baseline."""
    # Model architecture (matches MuZero for fair comparison)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    
    # Task configuration
    vocab_size: int = 2
    max_seq_len: int = 16
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Seq2SeqTransformer(nn.Module):
    """
    Transformer for Seq2Seq prediction.
    
    Uses encoder-decoder architecture with teacher forcing.
    """
    
    def __init__(self, config: SupervisedConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder_embed = nn.Embedding(config.vocab_size + 1, config.d_model)
        self.encoder_pos = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Decoder (autoregressive)
        self.decoder_embed = nn.Embedding(config.vocab_size + 1, config.d_model)  # +1 for start token
        self.decoder_pos = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Cross-attention layers
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout, batch_first=True)
            for _ in range(config.n_layers)
        ])
        self.cross_ln = nn.ModuleList([
            nn.LayerNorm(config.d_model) for _ in range(config.n_layers)
        ])
        
        # Output head
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode source sequence."""
        x = self.encoder_embed(src)
        x = self.encoder_pos(x)
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        return x
    
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
        """Decode with cross-attention to encoder output."""
        x = self.decoder_embed(tgt)
        x = self.decoder_pos(x)
        
        for i, layer in enumerate(self.decoder_layers):
            # Self-attention
            x = layer(x, mask=causal_mask)
            
            # Cross-attention
            attn_out, _ = self.cross_attention[i](x, memory, memory)
            x = self.cross_ln[i](x + attn_out)
        
        return x
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        
        Args:
            src: (batch, src_len) - input sequence
            tgt: (batch, tgt_len) - target sequence (shifted right for teacher forcing)
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        # Create causal mask
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=src.device) * float('-inf'),
            diagonal=1
        )
        
        # Encode and decode
        memory = self.encode(src)
        output = self.decode(tgt, memory, causal_mask)
        
        return self.output_head(output)
    
    def generate(self, src: torch.Tensor, max_len: int = None) -> torch.Tensor:
        """
        Autoregressive generation (without teacher forcing).
        
        Args:
            src: (batch, src_len) - input sequence
            max_len: Maximum output length
        Returns:
            output: (batch, max_len) - generated sequence
        """
        if max_len is None:
            max_len = src.size(1)
        
        batch_size = src.size(0)
        device = src.device
        
        # Encode
        memory = self.encode(src)
        
        # Start with start token (using vocab_size as start token id)
        output = torch.full((batch_size, 1), self.config.vocab_size, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            # Create causal mask
            tgt_len = output.size(1)
            causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device) * float('-inf'),
                diagonal=1
            )
            
            # Decode
            dec_output = self.decode(output, memory, causal_mask)
            
            # Get next token
            logits = self.output_head(dec_output[:, -1:])
            next_token = logits.argmax(dim=-1)
            
            output = torch.cat([output, next_token], dim=1)
        
        # Remove start token
        return output[:, 1:]


class SupervisedTrainer:
    """
    Training pipeline for supervised Seq2Seq baseline.
    """
    
    def __init__(
        self,
        task: str = "reversal",
        seq_length: int = 8,
        device: str = None
    ):
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Using device: {self.device}")
        
        # Environment (for generating data)
        self.env = create_env(task, seq_length)
        self.task = task
        self.seq_length = seq_length
        
        # Model configuration
        self.config = SupervisedConfig(
            vocab_size=self.env.vocab_size,
            max_seq_len=seq_length,
            device=device
        )
        
        # Create model
        self.model = Seq2SeqTransformer(self.config).to(device)
        
        # Training
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Statistics
        self.total_samples = 0
        self.losses: List[float] = []
        self.accuracies: List[float] = []
    
    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a batch of training data.
        
        Returns:
            src: Input sequences
            tgt_input: Target sequences (shifted right for teacher forcing)
            tgt_output: Target sequences (for loss computation)
        """
        src_list = []
        tgt_list = []
        
        for _ in range(batch_size):
            self.env.reset()
            src = self.env.input_seq.copy()
            tgt = self.env.target_seq.copy()
            
            src_list.append(src)
            tgt_list.append(tgt)
        
        src = torch.tensor(np.array(src_list), dtype=torch.long, device=self.device)
        tgt = torch.tensor(np.array(tgt_list), dtype=torch.long, device=self.device)
        
        # Create teacher forcing input (shift right, prepend start token)
        start_token = self.config.vocab_size
        tgt_input = torch.cat([
            torch.full((batch_size, 1), start_token, dtype=torch.long, device=self.device),
            tgt[:, :-1]
        ], dim=1)
        
        return src, tgt_input, tgt
    
    def train_step(self) -> Tuple[float, float]:
        """
        Perform one training step.
        
        Returns:
            loss, accuracy
        """
        self.model.train()
        
        # Generate batch
        src, tgt_input, tgt_output = self.generate_batch(self.config.batch_size)
        
        # Forward pass
        logits = self.model(src, tgt_input)
        
        # Compute loss
        loss = self.criterion(
            logits.view(-1, self.config.vocab_size),
            tgt_output.view(-1)
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == tgt_output).float().mean().item()
        
        self.total_samples += self.config.batch_size * self.seq_length
        
        return loss.item(), accuracy
    
    def evaluate(self, num_episodes: int = 100) -> float:
        """
        Evaluate the model using autoregressive generation.
        
        Args:
            num_episodes: Number of sequences to evaluate
        Returns:
            Average accuracy
        """
        self.model.eval()
        
        accuracies = []
        
        with torch.no_grad():
            for _ in range(num_episodes):
                self.env.reset()
                src = torch.tensor(self.env.input_seq, dtype=torch.long, device=self.device).unsqueeze(0)
                tgt = self.env.target_seq
                
                # Generate
                output = self.model.generate(src, self.seq_length)
                predictions = output.squeeze(0).cpu().numpy()
                
                # Compute accuracy
                accuracy = np.mean(predictions == tgt)
                accuracies.append(accuracy)
        
        return np.mean(accuracies)
    
    def train(
        self,
        max_samples: int = 10000,
        eval_interval: int = 100,
        target_accuracy: float = 0.99
    ) -> Dict:
        """
        Main training loop.
        
        Args:
            max_samples: Maximum number of samples
            eval_interval: How often to evaluate (in steps)
            target_accuracy: Stop when this accuracy is reached
        Returns:
            Training results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting Supervised Training")
        print(f"Task: {self.task}")
        print(f"Sequence length: {self.seq_length}")
        print(f"Max samples: {max_samples}")
        print(f"Target accuracy: {target_accuracy:.1%}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        best_accuracy = 0.0
        samples_to_target = None
        step = 0
        
        while self.total_samples < max_samples:
            loss, train_acc = self.train_step()
            self.losses.append(loss)
            self.accuracies.append(train_acc)
            step += 1
            
            # Logging
            if step % eval_interval == 0:
                eval_acc = self.evaluate(100)
                avg_loss = np.mean(self.losses[-100:])
                
                print(f"Step {step:5d} | "
                      f"Samples: {self.total_samples:6d} | "
                      f"Train Acc: {train_acc:.2%} | "
                      f"Eval Acc: {eval_acc:.2%} | "
                      f"Loss: {avg_loss:.4f}")
                
                if eval_acc > best_accuracy:
                    best_accuracy = eval_acc
                
                # Check if target reached
                if eval_acc >= target_accuracy and samples_to_target is None:
                    samples_to_target = self.total_samples
                    print(f"\n🎯 Reached {target_accuracy:.1%} accuracy at {samples_to_target} samples!")
        
        elapsed = time.time() - start_time
        final_acc = self.evaluate(100)
        
        results = {
            'total_samples': self.total_samples,
            'total_steps': step,
            'best_accuracy': best_accuracy,
            'final_accuracy': final_acc,
            'samples_to_target': samples_to_target,
            'elapsed_time': elapsed
        }
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total samples: {results['total_samples']}")
        print(f"Best accuracy: {results['best_accuracy']:.2%}")
        print(f"Final accuracy: {results['final_accuracy']:.2%}")
        print(f"Time: {elapsed:.1f}s")
        if samples_to_target:
            print(f"Samples to {target_accuracy:.1%}: {samples_to_target}")
        print(f"{'='*60}\n")
        
        return results
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


# Test code
if __name__ == "__main__":
    print("Testing Supervised Baseline...")
    
    trainer = SupervisedTrainer(
        task="reversal",
        seq_length=4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Quick training test
    results = trainer.train(max_samples=500, eval_interval=10)
    
    print(f"\nFinal evaluation: {trainer.evaluate(100):.2%}")
