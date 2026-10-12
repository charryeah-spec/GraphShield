import unittest

import numpy as np

from scripts.run_graphsage_oof import select_f1_threshold


class OOFRunnerTests(unittest.TestCase):
    def test_threshold_is_selected_only_from_supplied_inner_data(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.35, 0.8])
        threshold = select_f1_threshold(labels, probabilities)
        self.assertIn(threshold, probabilities)
        predictions = probabilities >= threshold
        self.assertGreater(predictions[labels == 1].sum(), 0)


if __name__ == "__main__":
    unittest.main()
