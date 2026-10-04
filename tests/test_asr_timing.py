"""Locks down the audio-to-frame mapping.

Regression: the MuseTalk prompt stride used to be a hardcoded ``index * 2``, which
is only correct at 25 fps. Running the pipeline at 10 fps therefore advanced the
audio features at 40% speed and the mouth barely moved.
"""

import unittest

from src.engine.asr import prompt_stride


class PromptStrideTest(unittest.TestCase):
    def test_25fps_matches_the_original_constant(self):
        self.assertEqual(prompt_stride(25), 2)

    def test_scales_with_a_lower_frame_rate(self):
        self.assertEqual(prompt_stride(10), 5)
        self.assertEqual(prompt_stride(12), 4)
        self.assertEqual(prompt_stride(50), 1)

    def test_never_returns_zero_for_high_frame_rates(self):
        self.assertEqual(prompt_stride(100), 1)
        self.assertEqual(prompt_stride(200), 1)

    def test_rejects_a_non_positive_frame_rate(self):
        with self.assertRaises(ValueError):
            prompt_stride(0)


if __name__ == "__main__":
    unittest.main()
