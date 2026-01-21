from dataclasses import dataclass
import math

@dataclass
class CurriculumSchedule:
    h_cycles: int
    halt_max_steps: int
    phase_name: str

class ProgressiveCurriculumScheduler:
    """
    Progressive curriculum for TRM training.

    Gradually increases difficulty by:
    1. Increasing H_cycles (thinking depth)
    2. Increasing halt_max_steps (maximum thinking time)
    """
    def __init__(
        self,
        total_steps: int,
        h_cycles_full: int = 3,
        halt_max_steps_full: int = 16,
        warmup_ratio: float = 0.2,
    ):
        self.total_steps = total_steps
        self.h_cycles_full = h_cycles_full
        self.halt_max_steps_full = halt_max_steps_full
        self.warmup_steps = int(total_steps * warmup_ratio)

    def get_schedule(self, step: int) -> CurriculumSchedule:
        if step < self.warmup_steps:
            # Linear ramp up
            progress = step / self.warmup_steps
            h_cycles = max(1, int(self.h_cycles_full * progress))
            halt_steps = max(2, int(self.halt_max_steps_full * progress))
            return CurriculumSchedule(h_cycles, halt_steps, "warmup")
        else:
            return CurriculumSchedule(self.h_cycles_full, self.halt_max_steps_full, "full")

    def get_expected_compute_savings(self) -> float:
        """Estimate compute savings compared to full training."""
        # Simple triangular approximation of savings during warmup
        return 0.5 * (self.warmup_steps / self.total_steps)
