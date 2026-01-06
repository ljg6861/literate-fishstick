"""
Training and evaluation for Universal Transformer Pointer Network.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time

from model import UniversalPointerNet, Config


class Trainer:
    """
    Trains the Universal Sorter on small lengths,
    then tests zero-shot on larger lengths.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = self.config.device
        
        # Create model
        self.model = UniversalPointerNet(self.config).to(self.device)
        
        # Compile model for speedups (PyTorch 2.0+)
        # mode='default' balances compile time vs run time
        # Compile model for speedups (PyTorch 2.0+)
        # mode='default' balances compile time vs run time
        if torch.cuda.device_count() > 1:
             print(f"⚡ Using {torch.cuda.device_count()} GPUs!")
             self.model = nn.DataParallel(self.model)
             self.raw_model = self.model.module
        else:
            self.raw_model = self.model
        
        # Compile model for speedups (PyTorch 2.0+)
        # mode='default' balances compile time vs run time
        # print("   Compiling model (torch.compile)...")
        # self.model = torch.compile(self.model)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=0.01
        )
        
        print(f"🚀 Universal Pointer Network")
        print(f"   Universal Block: {self.config.recurrent_steps}x recurrent")
        print(f"   Train: {self.config.train_lengths} → Test: 1000 (Zero-Shot!)")
        print(f"   Parameters: {self.raw_model.count_parameters():,}")
        
        self.step = 0
        self.samples = 0
        
        # Curriculum Setup
        self.curriculum_levels = [4, 8, 16, 32, 64] # Extend as needed
        self.active_level_idx = 0
        self.active_lengths = [self.curriculum_levels[0]]
        
        # History for all potential lengths
        self.acc_history = {n: deque(maxlen=50) for n in self.curriculum_levels}
        
        
        # Patience mechanism: force advance if stuck for too long
        self.steps_at_current_level = 0
        self.max_steps_per_level = 10000  # Force advance after 10k steps
        
    def check_curriculum(self):
        """Check if we should unlock the next length."""
        if self.active_level_idx >= len(self.curriculum_levels) - 1:
            return  # Max level reached
            
        current_max = self.curriculum_levels[self.active_level_idx]
        if not self.acc_history[current_max]:
             return

        current_acc = np.mean(self.acc_history[current_max])
        
        # Advance condition: High accuracy OR Patience exhausted
        advance = False
        reason = ""
        
        if current_acc >= 0.90:  # Lowered from 0.98 to 0.90
            advance = True
            reason = f"High Accuracy ({current_acc:.1%})"
        elif self.steps_at_current_level >= self.max_steps_per_level:
            advance = True
            reason = f"Patience Exhausted ({self.steps_at_current_level} steps)"
            
        if advance:
            self.active_level_idx += 1
            new_len = self.curriculum_levels[self.active_level_idx]
            self.active_lengths.append(new_len)
            self.steps_at_current_level = 0
            print(f"\n🚀 CURRICULUM LEVEL UP! Unlocked N={new_len} [{reason}]")
            
            # Reset history for the new length to avoid immediate promotion if it was accidentally high
            self.acc_history[new_len].clear()
    
    def train_step(self):
        """Train on all lengths simultaneously (Gradient Accumulation)."""
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss_val = 0.0
        total_correct = {}
        total_count = {}
        
        num_lengths = len(self.active_lengths)
        num_gpus = max(1, torch.cuda.device_count())
        
        # Train on currently active curriculum lengths
        for length in self.active_lengths:
            # Base capacity ref: N=64 -> B=48 seems safe (0.3GB/sample -> ~15GB)
            # Quadratic scaling because O(L^2) memory cost
            # N=4: 12,500 (capped at 8192)
            # N=8: 3,125
            # N=64: ~48
            safe_batch_per_gpu = 200000 // (length ** 2)
            safe_batch_per_gpu = max(2, min(8192, safe_batch_per_gpu))
            
            # Scale batch size by number of GPUs!
            safe_batch_total = safe_batch_per_gpu * num_gpus
            
            # Allow logical batch to go higher than config if capacity allows, 
            # Or usually config.samples is the target. 
            # Let's trust safe_batch_total as the primary driver for speed.
            batch_size = safe_batch_total
            # batch_size = max(1, min(self.config.samples_per_length * num_gpus, safe_batch_total))
            
            # === ALPHA ZERO UPDATE ===
            use_search = (self.step > 1000 and np.random.random() < 0.1) # 10% of batches after warmup
            
            if use_search:
                # Smaller batch for search because it's slow
                # Scale search batch relative to the SAFE dynamic batch size
                # Note: train_with_search is NOT currently parallelized, runs on main GPU.
                # So we use single-GPU safe batch size roughly.
                search_batch = max(2, safe_batch_per_gpu // 4)
                loss, correct_count = self.train_with_search(length, search_batch)
                
                # Normalize and backprop same as standard
                (loss / num_lengths).backward()
                total_loss_val += loss.item()
                total_correct[length] = correct_count
                total_count[length] = search_batch * length
                continue

            # Standard Training (Multi-GPU Enabled)
            
            # Generate batch of random numbers to sort (Continuous 0-1)
            x = torch.rand((batch_size, length), device=self.device)
            target_full = torch.argsort(x, dim=1) 
            
            # DataParallel Forward Pass
            # self.model(x, target_full) calls the new forward method in steps
            # DataParallel automatically splits x and target_full along dim 0
            # Returns: (loss_vector, correct_vector)
            step_losses, step_corrects = self.model(x, target_full)
            
            # Reduce results
            # loss is returned as vector of losses (one per GPU if DP, or scalar if not)
            loss = step_losses.mean() # Average loss across GPUs/Batch
            correct = step_corrects.sum().item() # Sum correct predictions
            
            # Combined loss
            # Note: The loss inside model is already the combined pointer+value loss
            
            # === GRADIENT ACCUMULATION ===
            # Normalize loss by number of tasks so gradients don't explode
            # Backpropagate IMMEDIATELY to free graph memory
            (loss / num_lengths).backward()
            
            total_loss_val += loss.item()
            total_correct[length] = correct
            total_count[length] = batch_size * length
        
        # Only step optimizer after all lengths processed
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Calculate average loss for reporting
        avg_loss = total_loss_val / num_lengths
        
        self.step += 1
        for n in self.active_lengths:
            acc = total_correct[n] / total_count[n]
            self.acc_history[n].append(acc)
            self.samples += total_count[n]
            
        self.steps_at_current_level += 1
        
        return avg_loss, {n: total_correct[n] / total_count[n] for n in self.active_lengths}
    
    @torch.no_grad()
    def evaluate(self, length: int, num_samples: int = 1000) -> float:
        """Evaluate on ANY length - zero-shot!"""
        self.model.eval()
        
        batch_size = min(100, num_samples)  # Smaller batches for long sequences
        
        # Use DataParallel for inference if available!
        if torch.cuda.device_count() > 1:
             # Scale batch size for multi-gpu eval
             batch_size *= torch.cuda.device_count()
        
        total_correct = 0
        total_elements = 0
        
        for _ in range(num_samples // batch_size + 1):
            current_batch = min(batch_size, num_samples - total_elements)
            if current_batch <= 0: break
            
            x = torch.rand((current_batch, length), device=self.device)
            target = torch.argsort(x, dim=1)
            
            # Inference Mode: forward(x) returns preds
            preds = self.model(x) # (B, L)
            
            # Calculate accuracy
            # Total matches
            matches = (preds == target).sum().item()
            
            total_correct += matches
            total_elements += current_batch * length
        
        return total_correct / total_elements if total_elements > 0 else 0.0
    
    def _compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence score for predictions.
        Uses negative entropy: high confidence = low entropy = high score.
        
        Args:
            logits: (B, L) raw pointer logits (may contain -inf for masked)
        Returns:
            confidence: (B,) confidence scores
        """
        # Softmax over valid positions only (masked are -inf)
        probs = F.softmax(logits, dim=-1)
        # Entropy: sum(-p * log(p)) 
        log_probs = torch.log(probs + 1e-9)
        entropy = -(probs * log_probs).sum(dim=-1)
        # Return negative entropy (higher = more confident)
        return -entropy
    
    @torch.no_grad()
    def evaluate_with_search(
        self, 
        length: int, 
        num_samples: int = 100,
        beam_width: int = 3,
        alpha: float = 0.65,
        lambda_value: float = 0.5
    ) -> float:
        """
        Evaluate with MCTS-style beam search for self-correction.
        
        Uses length-normalized scoring and value head for lookahead.
        
        Args:
            length: Sequence length to evaluate
            num_samples: Number of sequences to test
            beam_width: Number of parallel hypotheses to maintain
            alpha: Length normalization exponent (0.6-0.7 typical)
            lambda_value: Weight for value head predictions
        
        Returns:
            Accuracy (element-wise)
        """
        self.model.eval()
        
        total_correct = 0
        total_elements = 0
        
        for _ in range(num_samples):
            # Generate single sequence
            x = torch.rand((1, length), device=self.device)
            target = torch.argsort(x, dim=1).squeeze(0)  # (L,)
            
            # Beam state: list of (mask, raw_log_prob, predictions)
            # We store raw_log_prob and apply normalization when ranking
            beams = [(
                torch.zeros(1, length, device=self.device),
                0.0,  # raw cumulative log probability
                []
            )]
            
            for t in range(length):
                all_candidates = []
                
                for mask, raw_score, preds in beams:
                    # Encode current state
                    # USE RAW MODEL!
                    encoded = self.raw_model.encode(x, mask)
                    logits = self.raw_model.pointer(encoded, mask)
                    
                    # Get top candidates (beam_width or remaining positions)
                    remaining = (mask.squeeze(0) == 0).sum().item()
                    k = min(beam_width, remaining)
                    
                    if k == 0:
                        continue
                    
                    # Get full softmax over ALL remaining positions for proper probs
                    probs = F.softmax(logits.squeeze(0), dim=0)
                    topk_probs, topk_indices = torch.topk(probs, k)
                    
                    for i in range(k):
                        pos = topk_indices[i].item()
                        log_prob = torch.log(topk_probs[i] + 1e-9).item()
                        new_raw_score = raw_score + log_prob
                        
                        # Create new mask
                        new_mask = mask.clone()
                        new_mask[0, pos] = 1
                        
                        # Get value prediction for lookahead
                        # USE RAW MODEL!
                        new_encoded = self.raw_model.encode(x, new_mask)
                        value_pred = self.raw_model.value(new_encoded, new_mask).item()
                        
                        # Combined score: normalized log-prob + weighted value
                        step = t + 1
                        norm_score = new_raw_score / (step ** alpha)
                        combined_score = norm_score + lambda_value * value_pred
                        
                        all_candidates.append((
                            new_mask,
                            new_raw_score,  # Keep raw for future normalization
                            preds + [pos],
                            combined_score  # Used for ranking
                        ))
                
                # Keep top beam_width candidates by combined score
                all_candidates.sort(key=lambda x: x[3], reverse=True)
                beams = [(m, s, p) for m, s, p, _ in all_candidates[:beam_width]]
                
                if not beams:
                    break
            
            # Best beam
            if beams:
                best_preds = beams[0][2]
                # Count correct predictions
                for t, pred in enumerate(best_preds):
                    if pred == target[t].item():
                        total_correct += 1
                total_elements += length
            else:
                total_elements += length
        
        return total_correct / total_elements if total_elements > 0 else 0.0
    
    def train(self, max_steps: int = 50000, log_every: int = 2000):
        """Main training loop."""
        print(f"\n{'='*70}")
        print("Universal Transformer Training")
        print(f"Goal: Curriculum {self.curriculum_levels[0]} → {self.curriculum_levels[-1]}")
        print(f"{'='*70}\n")
        
        start = time.time()
        losses = deque(maxlen=100)
        
        while self.step < max_steps:
            loss, accs = self.train_step()
            losses.append(loss)
            
            # Check curriculum progress
            self.check_curriculum()
            
            if self.step % log_every == 0:
                elapsed = time.time() - start
                speed = self.samples / elapsed
                
                eval_accs = {n: self.evaluate(n, 500) for n in self.active_lengths}
                acc_str = " | ".join([f"N{n}:{eval_accs[n]:.0%}" for n in self.active_lengths])
                
                print(f"Step {self.step:5} | {acc_str} | Loss: {np.mean(losses):.4f} | {speed:,.0f}/s")
                
                if all(eval_accs[n] >= 0.95 for n in self.active_lengths) and self.active_level_idx == len(self.curriculum_levels) - 1:
                    print(f"\n🌟 Curriculum Completed! 95%+ on all lengths.")
                    break
        
        elapsed = time.time() - start
        print(f"\n{'='*70}")
        print(f"Training Complete! Time: {elapsed:.1f}s")
        print(f"{'='*70}")
    
    def zero_shot_eval(self):
        """The ULTIMATE test: zero-shot on unseen lengths!"""
        print("\n" + "="*70)
        print("🌟 ZERO-SHOT GENERALIZATION TEST")
        print("Trained on: " + ", ".join([f"N={n}" for n in self.config.train_lengths]))
        print("="*70)
        
        print("\n📊 Trained lengths:")
        for n in self.config.train_lengths:
            acc = self.evaluate(n, 1000)
            status = '✅' if acc >= 0.90 else ('🔶' if acc >= 0.80 else '❌')
            print(f"  N={n:4}: {acc:.2%} {status}")
        
        print("\n🚀 Zero-Shot OOD (Never trained!):")
        ood_lengths = [32, 48, 64, 100, 200, 500, 1000]
        for n in ood_lengths:
            # Fewer samples for very long sequences
            num_samples = 500 if n <= 100 else (200 if n <= 500 else 100)
            acc = self.evaluate(n, num_samples)
            
            if acc >= 0.90:
                status = '🌟 NEURAL ALGORITHM!'
            elif acc >= 0.70:
                status = '✅ Strong generalization'
            elif acc >= 0.50:
                status = '🔶 Partial'
            else:
                status = '❌'
            
            print(f"  N={n:4}: {acc:.2%} {status}")
        
        print("="*70)
    
    def save(self, path: str):
        """Save model checkpoint."""
        # Save RAW model state dict, not DataParallel wrapper
        torch.save({
            'model': self.raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'config': self.config,
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        # Load into RAW model
        self.raw_model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        print(f"Loaded checkpoint from {path} (step {self.step})")

    def train_with_search(self, length: int, batch_size: int):
        """
        AlphaZero-style training:
        1. Run Beam Search to find a good path.
        2. Use the path found by search as the POLICY target.
        3. Use the final result (sorted?) as the VALUE target.
        """
        # Generate data
        x = torch.rand((batch_size, length), device=self.device)
        target_sorted = torch.argsort(x, dim=1)
        
        pointer_loss = 0.0
        value_loss = 0.0
        correct_steps = 0
        
        # We process each item in the batch sequentially for search 
        # (since beam search isn't batched here yet)
        # To be efficient, we accumulate gradients locally? 
        # No, we can construct the batch of targets and run one forward pass.
        
        collected_states = [] # (x, mask)
        collected_policy_targets = [] # (target_idx)
        collected_value_targets = [] # (future_return)
        
        # Reuse evaluate_with_search logic but capture the trace
        # simpler beam search for training
        beam_width = 2 # Small beam for speed during training
        
        for b in range(batch_size):
            x_b = x[b:b+1] # (1, L)
            target_b = target_sorted[b] # (L)
            
            # Start Beam
            beams = [(torch.zeros(1, length, device=self.device), 0.0, [])] # (mask, score, path)
            
            # Run beam search
            for t in range(length):
                all_candidates = []
                for mask, score, path in beams:
                    # Expand
                    with torch.no_grad(): # No grad during search
                        # USE RAW MODEL
                        encoded = self.raw_model.encode(x_b, mask)
                        logits = self.raw_model.pointer(encoded, mask)
                        
                    # Top k
                    remaining = (mask.squeeze(0) == 0).sum().item()
                    k = min(beam_width, remaining)
                    if k == 0: continue
                    
                    probs = F.softmax(logits.squeeze(0), dim=0)
                    topk_probs, topk_indices = torch.topk(probs, k)
                    
                    for i in range(k):
                        pos = topk_indices[i].item()
                        log_p = torch.log(topk_probs[i] + 1e-9).item()
                        
                        # Value estimate
                        new_mask = mask.clone()
                        new_mask[0, pos] = 1
                        with torch.no_grad():
                            # USE RAW MODEL
                            new_enc = self.raw_model.encode(x_b, new_mask)
                            val = self.raw_model.value(new_enc, new_mask).item()
                            
                        new_score = score + log_p + val # Simple fusion
                        
                        all_candidates.append((new_mask, new_score, path + [pos]))
                
                # Prune
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                beams = all_candidates[:beam_width]
            
            # Select best path from beam
            best_beam = beams[0] # (mask, score, path)
            best_path = best_beam[2]
            
            # Check correctness
            # Calculate Return: 1.0 if perfectly sorted, else 0.0 (or partial?)
            # Let's use partial credit: fraction of correct positions
            # Actually, "is valid sort" is binary. 
            # But let's use the element-wise correctness as return z
            path_tensor = torch.tensor(best_path, device=self.device)
            # Correct if x[path[t]] == sorted_x[t]
            # Handle duplicates: check if values match
            x_vals = x_b[0, path_tensor]
            x_sorted_vals = x_b[0, target_b]
            
            # Future accuracy at each step is the return
            # Correctness of the REST of the sequence
            matches = (x_vals == x_sorted_vals).float()
            
            # Reconstruct training data from this trace
            curr_mask = torch.zeros(1, length, device=self.device)
            for t, action in enumerate(best_path):
                # State: (x_b, curr_mask)
                collected_states.append((x_b, curr_mask.clone()))
                
                # Policy Target: The action we actually took in the best path
                # Ideally we want a distribution over valid children in the beam,
                # but hardening to the best action is standard AlphaZero "target pi".
                collected_policy_targets.append(action)
                
                # Value Target: Mean accuracy of remaining steps
                # z_t = mean(matches[t:])
                z_t = matches[t:].mean().item() if t < length else 0.0
                collected_value_targets.append(z_t)
                
                if matches[t] == 1.0:
                    correct_steps += 1
                
                curr_mask[0, action] = 1
        
        # === BATCH UPDATE ===
        # Now we have a pile of (state, action, value) from the beams.
        # Run one big forward pass to train.
        
        if not collected_states:
            return torch.tensor(0.0, device=self.device), 0
            
        # Stack inputs
        b_x = torch.cat([s[0] for s in collected_states], dim=0) # (B*L, L)
        b_mask = torch.cat([s[1] for s in collected_states], dim=0) # (B*L, L)
        b_target_pi = torch.tensor(collected_policy_targets, device=self.device) # (B*L,)
        b_target_v = torch.tensor(collected_value_targets, device=self.device) # (B*L,)
        
        # Forward
        # USE RAW MODEL! (Or could use DP model if we pass through forward? No, we have specific loss here)
        # The 'train_with_search' loss is specific: CrossEntropy(logits) and MSE(value)
        # Our model.forward implements strict teacher-forced loop loss.
        # Here we have batched states.
        # So we use raw_model.encode/pointer/value directly.
        encoded = self.raw_model.encode(b_x, b_mask)
        logits = self.raw_model.pointer(encoded, b_mask)
        val_pred = self.raw_model.value(encoded, b_mask).squeeze(-1)
        
        # Policy Loss: CrossEntropy against the search-selected action
        # Note: We rely on the search to have picked a GOOD action.
        # If search failed, we might reinforce bad actions? 
        # AlphaZero usually relies on MCTS being an improvement op.
        # Given we used beam search, it should be decent.
        # Also, we can filter to only train on Correct steps? 
        # "Filter label noise": Only train if z_t > threshold?
        # User said: "Train pointer logits to put probability mass over all valid actions" (Step 1)
        # Step 2 implies aligning with search.
        # Let's trust the search for now.
        
        p_loss = F.cross_entropy(logits, b_target_pi)
        v_loss = F.mse_loss(val_pred, b_target_v)
        
        loss = p_loss + 0.1 * v_loss
        
        return loss, correct_steps
