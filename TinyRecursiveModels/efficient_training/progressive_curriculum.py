"""
Progressive Training Curriculum for TRM.

This module provides a scheduler for dynamically adjusting H_cycles and halt_max_steps
during training to achieve ~47% compute reduction while maintaining final accuracy.

Theory:
- Early training: Model learns basic pattern matching (shallow computation sufficient)
- Later training: Model refines with full capacity (deep computation needed)
- Progressive exposure: Smooth transition prevents "capacity shock"
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class ProgressiveSchedule:
    """Current schedule values."""
    h_cycles: int
    halt_max_steps: int
    phase: int
    phase_name: str
    progress: float


class ProgressiveCurriculumScheduler:
    """
    Scheduler for progressive H_cycles and halt_max_steps curriculum.
    
    Uses smooth ramping between phases to avoid "capacity shock":
    - Phase 1: Low capacity (H=1, T=4) - fast initial learning
    - Phase 2: Medium capacity (H=2, T=8) - intermediate refinement
    - Phase 3: Full capacity (H=3, T=16) - final training
    
    Compute reduction: ~47% compared to full capacity throughout.
    """
    
    def __init__(
        self,
        total_steps: int,
        # Phase boundaries (as fractions of total training)
        phase1_end: float = 0.3,   # 0-30%: shallow
        phase2_end: float = 0.6,   # 30-60%: medium  
        # Transition windows (steps to smooth transition)
        transition_steps: int = 500,
        # Target values
        h_cycles_full: int = 3,
        halt_max_steps_full: int = 16,
    ):
        self.total_steps = total_steps
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.transition_steps = transition_steps
        self.h_cycles_full = h_cycles_full
        self.halt_max_steps_full = halt_max_steps_full
        
        # Generate phase configurations dynamically based on full capacity
        # Phase 1: ~33% of h_cycles, ~25% of halt_steps (shallow)
        # Phase 2: ~67% of h_cycles, ~50% of halt_steps (medium)
        # Phase 3: 100% of h_cycles, 100% of halt_steps (full)
        h1 = max(1, h_cycles_full // 3)  # At least 1
        h2 = max(1, (2 * h_cycles_full) // 3)  # 2/3
        h3 = h_cycles_full
        
        t1 = max(4, halt_max_steps_full // 4)  # At least 4
        t2 = max(4, halt_max_steps_full // 2)  # 1/2
        t3 = halt_max_steps_full
        
        self.phase_configs = [
            (h1, t1),   # Phase 1: shallow
            (h2, t2),   # Phase 2: medium
            (h3, t3),   # Phase 3: full
        ]
        
        # Compute phase boundaries in steps
        self.phase1_step = int(phase1_end * total_steps)
        self.phase2_step = int(phase2_end * total_steps)
    
    def get_schedule(self, step: int) -> ProgressiveSchedule:
        """
        Get current schedule values for the given training step.
        
        Args:
            step: Current training step (0-indexed)
            
        Returns:
            ProgressiveSchedule with current h_cycles, halt_max_steps, and metadata
        """
        progress = step / max(1, self.total_steps)
        
        if step < self.phase1_step:
            # Phase 1: Shallow
            h_cycles, halt_max_steps = self.phase_configs[0]
            phase, phase_name = 1, "shallow"
            
            # Check for transition to phase 2
            if step >= self.phase1_step - self.transition_steps:
                # Smooth transition
                t = (step - (self.phase1_step - self.transition_steps)) / self.transition_steps
                h_cycles, halt_max_steps = self._interpolate(
                    self.phase_configs[0], 
                    self.phase_configs[1], 
                    t
                )
                phase_name = "shallow→medium"
                
        elif step < self.phase2_step:
            # Phase 2: Medium
            h_cycles, halt_max_steps = self.phase_configs[1]
            phase, phase_name = 2, "medium"
            
            # Check for transition to phase 3
            if step >= self.phase2_step - self.transition_steps:
                t = (step - (self.phase2_step - self.transition_steps)) / self.transition_steps
                h_cycles, halt_max_steps = self._interpolate(
                    self.phase_configs[1],
                    self.phase_configs[2],
                    t
                )
                phase_name = "medium→full"
                
        else:
            # Phase 3: Full capacity
            h_cycles, halt_max_steps = self.phase_configs[2]
            phase, phase_name = 3, "full"
        
        return ProgressiveSchedule(
            h_cycles=h_cycles,
            halt_max_steps=halt_max_steps,
            phase=phase,
            phase_name=phase_name,
            progress=progress,
        )
    
    def _interpolate(
        self, 
        config1: Tuple[int, int], 
        config2: Tuple[int, int], 
        t: float
    ) -> Tuple[int, int]:
        """Smoothly interpolate between two configs using cosine schedule."""
        # Cosine interpolation for smoother transition
        t = 0.5 * (1 - math.cos(math.pi * t))
        
        h1, s1 = config1
        h2, s2 = config2
        
        # For H_cycles, we use ceiling to ensure we don't go below target
        h = int(math.ceil(h1 + t * (h2 - h1)))
        # For halt_steps, round to nearest
        s = int(round(s1 + t * (s2 - s1)))
        
        return h, s
    
    def get_expected_compute_savings(self) -> float:
        """
        Calculate expected compute savings compared to full capacity.
        
        Returns:
            Fraction of compute saved (0.0 = no savings, 1.0 = all saved)
        """
        # Get phase configs
        h1, t1 = self.phase_configs[0]
        h2, t2 = self.phase_configs[1]
        h3, t3 = self.phase_configs[2]
        
        # Calculate relative cost of each phase vs full
        phase1_cost = (h1 / h3) * (t1 / t3)
        phase2_cost = (h2 / h3) * (t2 / t3)
        phase3_cost = 1.0  # Full capacity
        
        weighted_cost = (
            self.phase1_end * phase1_cost +
            (self.phase2_end - self.phase1_end) * phase2_cost +
            (1.0 - self.phase2_end) * phase3_cost
        )
        
        return 1.0 - weighted_cost
    
    def __repr__(self) -> str:
        savings = self.get_expected_compute_savings()
        return (
            f"ProgressiveCurriculumScheduler(\n"
            f"  total_steps={self.total_steps},\n"
            f"  phases=[0-{self.phase1_step}]=shallow, "
            f"[{self.phase1_step}-{self.phase2_step}]=medium, "
            f"[{self.phase2_step}+]=full,\n"
            f"  expected_savings={savings:.1%}\n"
            f")"
        )


def create_default_scheduler(total_steps: int) -> ProgressiveCurriculumScheduler:
    """Create scheduler with recommended settings."""
    return ProgressiveCurriculumScheduler(
        total_steps=total_steps,
        phase1_end=0.3,
        phase2_end=0.6,
        transition_steps=min(500, total_steps // 20),  # 5% of phase length
    )


# Test
if __name__ == "__main__":
    scheduler = create_default_scheduler(total_steps=31000)
    print(scheduler)
    print(f"\nExpected compute savings: {scheduler.get_expected_compute_savings():.1%}")
    
    # Sample schedule at various points
    test_steps = [0, 5000, 9000, 10000, 15000, 18000, 20000, 25000, 30000]
    print("\nSchedule samples:")
    for step in test_steps:
        s = scheduler.get_schedule(step)
        print(f"  step={step:5d}: H={s.h_cycles}, T={s.halt_max_steps:2d}, phase={s.phase_name}")
