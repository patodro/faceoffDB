import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_baseline import extract_flow_features, load_records


class LoadRecordsTests(unittest.TestCase):
    def test_resolves_raw_and_curated_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotations = root / "annotations.csv"
            clips_root = root / "raw_clips"
            reference_root = root / "reference_library"
            (clips_root / "game-1").mkdir(parents=True)
            (reference_root / "clean_examples").mkdir(parents=True)
            (reference_root / "dismissal_examples").mkdir(parents=True)
            (reference_root / "reset_examples").mkdir(parents=True)
            (clips_root / "game-1" / "clean.mp4").touch()
            (clips_root / "game-1" / "raw-fallback.mp4").touch()
            (reference_root / "clean_examples" / "clean.mp4").touch()
            (reference_root / "dismissal_examples" / "dismissal.mp4").touch()
            (reference_root / "clean_examples" / "reset.mp4").touch()
            with annotations.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=("game_id", "clip_filename", "linesman_id", "label"),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "game_id": "game-1",
                            "clip_filename": "clean",
                            "linesman_id": "L1",
                            "label": "CLEAN",
                        },
                        {
                            "game_id": "game-1",
                            "clip_filename": "dismissal",
                            "linesman_id": "L1",
                            "label": "DISMISSAL",
                        },
                        {
                            "game_id": "game-1",
                            "clip_filename": "reset",
                            "linesman_id": "L2",
                            "label": "RESET",
                        },
                        {
                            "game_id": "game-1",
                            "clip_filename": "raw-fallback",
                            "linesman_id": "L2",
                            "label": "CLEAN",
                        },
                    ]
                )

            records = load_records(annotations, clips_root, reference_root)

            self.assertEqual(len(records), 4)
            self.assertEqual(
                records[0].path,
                reference_root / "clean_examples" / "clean.mp4",
            )
            self.assertEqual(
                records[1].path,
                reference_root / "dismissal_examples" / "dismissal.mp4",
            )
            self.assertEqual(
                records[2].path,
                reference_root / "clean_examples" / "reset.mp4",
            )
            self.assertEqual(
                records[3].path,
                clips_root / "game-1" / "raw-fallback.mp4",
            )


class OpticalFlowFeatureTests(unittest.TestCase):
    def test_features_are_finite_and_fixed_length(self) -> None:
        frames = []
        for offset in range(8):
            frame = np.zeros((32, 48), dtype=np.float32)
            frame[8:16, 4 + offset : 12 + offset] = 1.0
            frames.append(frame)

        features = extract_flow_features(frames)

        self.assertEqual(features.shape, (160,))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
