import unittest
from types import SimpleNamespace

import numpy as np

from utils.data_processor import DataProcessor


class FastModeTests(unittest.TestCase):
    def test_fast_mode_keeps_pattern_dependencies(self):
        config = SimpleNamespace(
            NORMAL_CLASS=0,
            FRAUD_CLASS=1,
            FAST_MODE=True,
            USE_TIME_FEATURES=False,
        )
        raw = {
            "x": np.array(
                [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]],
                dtype=np.float64,
            ),
            "y": np.array([0, 1, 0, -100]),
            "edge_index": np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
            "edge_type": np.array([0, 0, 0], dtype=np.int64),
            "train_mask": np.array([0, 1, 2], dtype=np.int64),
            "test_mask": np.array([3], dtype=np.int64),
        }
        features = DataProcessor(config).create_enhanced_features(raw)
        self.assertEqual(features.shape[0], 4)
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
