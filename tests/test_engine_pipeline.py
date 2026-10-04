import pickle
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from src.engine import LipSyncInput, MuseTalk, Wav2Lip


class EnginePipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        full_dir = self.root / "full_imgs"
        face_dir = self.root / "face_imgs"
        full_dir.mkdir()
        face_dir.mkdir()
        for index in range(2):
            full = np.full((300, 300, 3), index * 20, dtype=np.uint8)
            face = np.full((256, 256, 3), 100 + index * 20, dtype=np.uint8)
            cv2.imwrite(str(full_dir / f"{index:08d}.png"), full)
            cv2.imwrite(str(face_dir / f"{index:08d}.png"), face)
        (self.root / "coords.pkl").write_bytes(pickle.dumps([(10, 266, 20, 276)] * 2))
        torch.save(torch.zeros(2, 8, 32, 32), self.root / "latens.pt")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_musetalk_pipeline(self):
        engine = MuseTalk(lambda faces, features, latents: faces)
        output = engine.process(LipSyncInput(self.root, audio_features=np.zeros((3, 4))))
        self.assertEqual(output.frame_count, 3)
        self.assertEqual(output.frames[0].shape, (300, 300, 3))

    def test_wav2lip_pipeline(self):
        engine = Wav2Lip(lambda mel, images: images * 255)
        output_path = self.root / "result.mp4"
        output = engine.process(LipSyncInput(self.root, audio=np.zeros(16000), frame_count=2, output_path=output_path))
        self.assertEqual(output.frame_count, 2)
        self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
