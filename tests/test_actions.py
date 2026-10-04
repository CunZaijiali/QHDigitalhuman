import pickle
import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.domain.actions import (
    ActionCache,
    ActionRef,
    list_actions,
    parse_action_name,
    weighted_choice,
)
from src.session.transition import ActionTransition


class ParseActionNameTest(unittest.TestCase):
    def test_plain_name_gets_default_weight(self):
        self.assertEqual(parse_action_name("NOD"), ("NOD", 1.0))

    def test_trailing_number_is_the_weight(self):
        self.assertEqual(parse_action_name("NOD-3"), ("NOD", 3.0))
        self.assertEqual(parse_action_name("NOD-1.5"), ("NOD", 1.5))

    def test_inner_hyphen_is_part_of_the_name(self):
        self.assertEqual(parse_action_name("WAVE-HAND"), ("WAVE-HAND", 1.0))
        self.assertEqual(parse_action_name("WAVE-HAND-2"), ("WAVE-HAND", 2.0))

    def test_zero_weight_keeps_the_literal_name(self):
        self.assertEqual(parse_action_name("NOD-0"), ("NOD-0", 1.0))


def _ref(name, weight):
    return ActionRef(
        behavior="IDLE",
        action=name,
        weight=weight,
        dir=Path("x") / name,
        engine="musetalkv15",
        root=Path("x") / name / "musetalkv15",
    )


class WeightedChoiceTest(unittest.TestCase):
    def test_uniform_when_all_weights_are_zero(self):
        rng = random.Random(0)
        refs = [_ref("A", 0.0), _ref("B", 0.0)]
        picks = {weighted_choice(refs, rng).action for _ in range(50)}
        self.assertEqual(picks, {"A", "B"})

    def test_heavier_action_dominates(self):
        rng = random.Random(7)
        refs = [_ref("LIGHT", 1.0), _ref("HEAVY", 5.0)]
        counts = {"LIGHT": 0, "HEAVY": 0}
        for _ in range(2000):
            counts[weighted_choice(refs, rng).action] += 1
        self.assertGreater(counts["HEAVY"], counts["LIGHT"] * 2)


class ActionTransitionTest(unittest.TestCase):
    def test_frame_count_from_seconds_and_fps(self):
        self.assertEqual(ActionTransition(0.2, 25).frames, 5)
        self.assertEqual(ActionTransition(0.0, 25).frames, 0)

    def test_inactive_without_a_source_frame(self):
        transition = ActionTransition(0.2, 25)
        transition.arm(None)
        self.assertFalse(transition.active)
        frame = np.full((4, 4, 3), 10, np.uint8)
        self.assertIs(transition.apply(frame), frame)

    def test_blends_towards_the_target_then_stops(self):
        transition = ActionTransition(0.2, 25)
        source = np.zeros((4, 4, 3), np.uint8)
        target = np.full((4, 4, 3), 100, np.uint8)
        transition.arm(source)

        first = transition.apply(target)
        self.assertGreater(int(first.mean()), 0)
        self.assertLess(int(first.mean()), 100)

        last = None
        while transition.active:
            last = transition.apply(target)
        self.assertIsNotNone(last)
        # Smoothstep ends exactly on the target.
        self.assertTrue(np.allclose(last, target, atol=1))

    def test_resizes_a_mismatched_source(self):
        transition = ActionTransition(0.2, 25)
        transition.arm(np.zeros((8, 8, 3), np.uint8))
        out = transition.apply(np.full((4, 4, 3), 50, np.uint8))
        self.assertEqual(out.shape, (4, 4, 3))

    def test_cancel_stops_blending(self):
        transition = ActionTransition(0.2, 25)
        transition.arm(np.zeros((4, 4, 3), np.uint8))
        transition.cancel()
        target = np.full((4, 4, 3), 50, np.uint8)
        self.assertTrue(np.allclose(transition.apply(target), target))


class ActionCatalogTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for name in ("NOD", "WAVEHAND-5", "BROKEN"):
            action = self.root / "IDLE" / name
            engine = action / "musetalkv15"
            (engine / "full_imgs").mkdir(parents=True)
            (engine / "face_imgs").mkdir(parents=True)
            cv2.imwrite(str(engine / "full_imgs" / "00000000.png"), np.zeros((8, 8, 3), np.uint8))
            cv2.imwrite(str(engine / "face_imgs" / "00000000.png"), np.zeros((4, 4, 3), np.uint8))
            if name != "BROKEN":  # BROKEN has no coords.pkl and must be skipped
                (engine / "coords.pkl").write_bytes(pickle.dumps([(0, 4, 0, 4)]))

    def tearDown(self):
        self._tmp.cleanup()

    def test_lists_usable_actions_with_weights(self):
        refs = list_actions(self.root / "IDLE", engine="musetalkv15")
        self.assertEqual([(r.action, r.weight) for r in refs], [("NOD", 1.0), ("WAVEHAND", 5.0)])

    def test_skips_actions_without_the_engine_assets(self):
        refs = list_actions(self.root / "IDLE", engine="musetalkv15")
        self.assertNotIn("BROKEN", [r.action for r in refs])


class ActionCacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.roots = []
        for name in ("A", "B", "C"):
            engine = self.root / name / "musetalkv15"
            (engine / "full_imgs").mkdir(parents=True)
            (engine / "face_imgs").mkdir(parents=True)
            for index in range(2):
                frame = np.full((6, 6, 3), index * 10, np.uint8)
                cv2.imwrite(str(engine / "full_imgs" / f"{index:08d}.png"), frame)
                cv2.imwrite(str(engine / "face_imgs" / f"{index:08d}.png"), frame)
            (engine / "coords.pkl").write_bytes(pickle.dumps([(0, 6, 0, 6)] * 2))
            self.roots.append(engine)

    def tearDown(self):
        self._tmp.cleanup()

    def test_rejects_unknown_strategy(self):
        with self.assertRaises(ValueError):
            ActionCache(strategy="sometimes")

    def test_preloads_frames_on_first_get(self):
        cache = ActionCache(limit=2)
        avatar = cache.get(self.roots[0])
        self.assertTrue(avatar.preloaded)
        self.assertGreater(avatar.loaded_bytes, 0)

    def test_evicts_least_recently_used(self):
        cache = ActionCache(limit=1)
        first = cache.get(self.roots[0])
        cache.get(self.roots[1])
        self.assertFalse(first.preloaded)  # evicted -> released
        self.assertEqual(cache.stats()["cached"], 1)

    def test_second_get_is_a_hit(self):
        cache = ActionCache(limit=2)
        a = cache.get(self.roots[0])
        b = cache.get(self.roots[0])
        self.assertIs(a, b)
        self.assertEqual(cache.stats()["hits"], 1)

    def test_eager_strategy_keeps_everything(self):
        cache = ActionCache(limit=1, strategy="eager")
        cache.set_total(len(self.roots))
        for root in self.roots:
            cache.get(root)
        self.assertEqual(cache.stats()["cached"], 3)


if __name__ == "__main__":
    unittest.main()
