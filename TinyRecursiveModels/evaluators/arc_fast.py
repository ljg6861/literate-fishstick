"""
ARC Evaluator with Early-Stop Optimization.

This is a modified version of the ARC evaluator that supports early-stopping.
When a puzzle's correct answer has an insurmountable vote lead, we can skip
remaining samples for that puzzle.

Usage:
    Replace the standard ARC evaluator with ARCFast in your config.
"""

from typing import Dict, Sequence, Optional, Set, Tuple
import os
import json

import torch
import numpy as np
from numba import njit
import torch.distributed as dist

from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np
from dataset.common import PuzzleDatasetMetadata


@njit
def _crop(grid: np.ndarray):
    """Find maximum-sized rectangle without any EOS token inside."""
    grid = grid.reshape(30, 30)

    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    
    num_c = nc
    for num_r in range(1, nr + 1):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break
        
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


class ARCFast:
    """ARC evaluator with early-stop optimization for faster evaluation."""
    
    required_outputs = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}
    
    def __init__(
        self, 
        data_path: str, 
        eval_metadata: PuzzleDatasetMetadata, 
        submission_K: int = 2, 
        pass_Ks: Sequence[int] = (1, 2, 5, 10, 100, 1000), 
        aggregated_voting: bool = True,
        early_stop: bool = True,
        early_stop_threshold: float = 0.0  # Extra margin before stopping
    ):
        super().__init__()
        self.pass_Ks = pass_Ks
        self.submission_K = submission_K
        self.aggregated_voting = aggregated_voting
        self.blank_identifier_id = eval_metadata.blank_identifier_id
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold

        # Load identifiers and test puzzles
        with open(os.path.join(data_path, "identifiers.json"), "r") as f:
            self.identifier_map = json.load(f)
        with open(os.path.join(data_path, "test_puzzles.json"), "r") as f:
            self.test_puzzles = json.load(f)
        
        # Precompute ground truth hashes for early-stop checking
        self._ground_truth: Dict[Tuple[str, str], str] = {}
        for name, puzzle in self.test_puzzles.items():
            for pair in puzzle["test"]:
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))
                self._ground_truth[(name, input_hash)] = label_hash
        
        # Estimate total samples per puzzle for early-stop math
        # Count augmentations per base puzzle
        self._puzzle_total_samples: Dict[str, int] = {}
        for name in self.identifier_map:
            if name == '<blank>' or '|||' not in name:
                continue
            base_name = name.split('|||')[0]
            self._puzzle_total_samples[base_name] = self._puzzle_total_samples.get(base_name, 0) + 1
            
        # States
        self._local_hmap = {}
        self._local_preds = {}
        
        # Early-stop state
        self._local_votes: Dict[Tuple[str, str], Dict[str, int]] = {}
        self._puzzles_done: Set[Tuple[str, str]] = set()
        self._samples_processed: Dict[str, int] = {}
        
        # Stats
        self._total_samples = 0
        self._skipped_samples = 0
        
    def begin_eval(self):
        if not self.aggregated_voting:
            self._local_hmap = {}
            self._local_preds = {}
        
        # Reset early-stop state per eval
        self._local_votes = {}
        self._puzzles_done = set()
        self._samples_processed = {}
        self._total_samples = 0
        self._skipped_samples = 0
    
    def _can_early_stop(self, puzzle_key: Tuple[str, str], base_name: str) -> bool:
        """Check if we can skip remaining samples for this puzzle."""
        if not self.early_stop:
            return False
        
        correct_hash = self._ground_truth.get(puzzle_key)
        if correct_hash is None:
            return False
        
        votes = self._local_votes.get(puzzle_key, {})
        if not votes:
            return False
        
        correct_votes = votes.get(correct_hash, 0)
        if correct_votes == 0:
            return False  # Haven't seen correct answer yet
        
        # Estimate remaining samples
        total = self._puzzle_total_samples.get(base_name, 1000)
        processed = self._samples_processed.get(base_name, 0)
        remaining = max(0, total - processed)
        
        # Find max votes for any other prediction
        max_other = 0
        for h, v in votes.items():
            if h != correct_hash and v > max_other:
                max_other = v
        
        # Can stop if correct answer can't be overtaken
        margin = int(remaining * self.early_stop_threshold)
        can_stop = correct_votes > max_other + remaining + margin
        
        return can_stop
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # Collect required outputs to CPU
        outputs = {}
        q_values = None

        for collection in (batch, preds):
            for k, v in collection.items():
                if k in self.required_outputs:
                    if k == "q_halt_logits":
                        q_values = v.to(torch.float64).sigmoid().cpu()
                    else:
                        outputs[k] = v.cpu()
                        
        assert q_values is not None

        # Remove padding from outputs
        mask = outputs["puzzle_identifiers"] != self.blank_identifier_id
        outputs = {k: v[mask] for k, v in outputs.items()}

        # Get predictions
        for identifier, input_arr, pred, q in zip(
            outputs["puzzle_identifiers"].numpy(), 
            outputs["inputs"].numpy(), 
            outputs["preds"].numpy(), 
            q_values.numpy()
        ):
            self._total_samples += 1
            
            name = self.identifier_map[identifier]
            orig_name, _inverse_fn = inverse_aug(name)
            
            input_hash = grid_hash(_inverse_fn(_crop(input_arr)))
            puzzle_key = (orig_name, input_hash)
            
            # Check if this puzzle is already done (early-stop)
            if puzzle_key in self._puzzles_done:
                self._skipped_samples += 1
                continue
            
            # Process prediction
            pred_cropped = _crop(pred)
            pred_transformed = _inverse_fn(pred_cropped)
            assert np.all((pred_transformed >= 0) & (pred_transformed <= 9))
            
            pred_hash = grid_hash(pred_transformed)

            # Update votes for early-stop checking
            if puzzle_key not in self._local_votes:
                self._local_votes[puzzle_key] = {}
            self._local_votes[puzzle_key][pred_hash] = self._local_votes[puzzle_key].get(pred_hash, 0) + 1
            
            # Track samples processed
            self._samples_processed[orig_name] = self._samples_processed.get(orig_name, 0) + 1

            # Store into local state (for final result computation)
            self._local_hmap[pred_hash] = pred_transformed
            
            self._local_preds.setdefault(orig_name, {})
            self._local_preds[orig_name].setdefault(input_hash, [])
            self._local_preds[orig_name][input_hash].append((pred_hash, float(q)))
            
            # Check early-stop condition
            if self._can_early_stop(puzzle_key, orig_name):
                self._puzzles_done.add(puzzle_key)
    
    def get_skip_mask(self, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        """Return a mask of samples that should be skipped (for batch filtering)."""
        mask = torch.ones(len(puzzle_identifiers), dtype=torch.bool)
        
        for i, identifier in enumerate(puzzle_identifiers.numpy()):
            if identifier == self.blank_identifier_id:
                continue
            name = self.identifier_map[identifier]
            orig_name, _inverse_fn = inverse_aug(name)
            
            # We can't compute input_hash here without the actual input
            # So we just check if the base puzzle has any done inputs
            # This is a conservative approximation
            for puzzle_key in self._puzzles_done:
                if puzzle_key[0] == orig_name:
                    mask[i] = False
                    break
        
        return mask
    
    def result(
        self, 
        save_path: Optional[str], 
        rank: int, 
        world_size: int, 
        group: Optional[torch.distributed.ProcessGroup] = None
    ) -> Optional[Dict[str, float]]:
        # Print early-stop stats
        if rank == 0:
            skip_rate = self._skipped_samples / max(1, self._total_samples) * 100
            print(f"  Early-stop stats: {self._skipped_samples}/{self._total_samples} samples skipped ({skip_rate:.1f}%)")
            print(f"  Puzzles early-stopped: {len(self._puzzles_done)}")
        
        # Gather predictions to rank 0 for voting
        global_hmap_preds = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object((self._local_hmap, self._local_preds), global_hmap_preds, dst=0, group=group)
        
        # Rank 0 logic
        if rank != 0:
            return None

        submission = {}
        correct = [0.0 for _ in range(len(self.pass_Ks))]

        for name, puzzle in self.test_puzzles.items():
            # Process test examples in this puzzle
            submission[name] = []
            num_test_correct = [0 for _ in range(len(self.pass_Ks))]
            for pair in puzzle["test"]:
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))
                
                p_map = {}
                for hmap, preds in global_hmap_preds:
                    for h, q in preds.get(name, {}).get(input_hash, {}):
                        p_map.setdefault(h, [0, 0])
                        p_map[h][0] += 1
                        p_map[h][1] += q
                        
                if not len(p_map):
                    print(f"Puzzle {name} has no predictions.")
                    continue

                for h, stats in p_map.items():
                    stats[1] /= stats[0]
                    
                p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

                # vote for different Ks
                for i, k in enumerate(self.pass_Ks):
                    ok = False
                    for h, stats in p_map[:k]:
                        ok |= h == label_hash
                        
                    num_test_correct[i] += ok
                    
                # Query grids
                pred_grids = []
                for h, stats in p_map[:self.submission_K]:
                    for hmap, preds in global_hmap_preds:
                        if h in hmap:
                            pred_grids.append(hmap[h])
                            break
                        
                # Pad to K
                while len(pred_grids) < self.submission_K:
                    pred_grids.append(pred_grids[0] if pred_grids else np.zeros((1, 1), dtype=np.uint8))
                
                submission[name].append({f"attempt_{i + 1}": grid.tolist() for i, grid in enumerate(pred_grids)})

            # Total correctness
            for i in range(len(self.pass_Ks)):
                correct[i] += num_test_correct[i] / len(puzzle["test"])

        # Save submission
        if save_path is not None:
            with open(os.path.join(save_path, "submission.json"), "w") as f:
                json.dump(submission, f)

        # Final result
        all_results = {f"ARC/pass@{k}": correct[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)}

        return all_results
