
import unittest
import torch
from efficient_training.progressive_curriculum import ProgressiveCurriculumScheduler, CurriculumSchedule

class TestProgressiveCurriculum(unittest.TestCase):
    def test_curriculum_schedule(self):
        scheduler = ProgressiveCurriculumScheduler(
            total_steps=1000,
            h_cycles_full=4,
            halt_max_steps_full=20,
            warmup_ratio=0.5
        )

        # Test start
        schedule_start = scheduler.get_schedule(0)
        self.assertEqual(schedule_start.phase_name, "warmup")
        self.assertLess(schedule_start.h_cycles, 4)

        # Test middle
        schedule_mid = scheduler.get_schedule(250)
        self.assertEqual(schedule_mid.phase_name, "warmup")

        # Test full
        schedule_full = scheduler.get_schedule(600)
        self.assertEqual(schedule_full.phase_name, "full")
        self.assertEqual(schedule_full.h_cycles, 4)
        self.assertEqual(schedule_full.halt_max_steps, 20)

    def test_savings_calculation(self):
        scheduler = ProgressiveCurriculumScheduler(100, warmup_ratio=0.5)
        savings = scheduler.get_expected_compute_savings()
        self.assertGreater(savings, 0)
        self.assertLess(savings, 1)

if __name__ == '__main__':
    unittest.main()
