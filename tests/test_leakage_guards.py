import unittest

import numpy as np

from utils.leakage_audit import best_orientation_auc, roc_auc
from utils.split_guard import assert_disjoint, temporal_holdout
from utils.temporal_features import (
    ExplicitTemporalFeatureBuilder,
    TemporalFeatureSpec,
    edge_order_permutation_change,
)


class LeakageGuardTests(unittest.TestCase):
    def test_auc_detects_reverse_order_proxy(self):
        labels = np.array([1, 1, 0, 0])
        node_ids = np.arange(4)
        self.assertEqual(roc_auc(labels, node_ids), 0.0)
        self.assertEqual(best_orientation_auc(labels, node_ids), 1.0)

    def test_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Evaluation leakage"):
            assert_disjoint(np.array([1, 2]), np.array([2, 3]))

    def test_temporal_holdout_is_forward_only(self):
        indices = np.arange(10)
        timestamps = np.array([4, 2, 8, 0, 1, 9, 7, 3, 6, 5])
        train, validation = temporal_holdout(indices, timestamps, 0.2, 0.1)
        self.assertLess(timestamps[train].max(), timestamps[validation].min())
        self.assertEqual(len(validation), 2)

    def test_explicit_builder_uses_training_bounds(self):
        x = np.array([[10.0], [20.0], [30.0]])
        builder = ExplicitTemporalFeatureBuilder(
            [TemporalFeatureSpec(0, "event_time", "Unix seconds from source system")]
        )
        features, names = builder.fit_transform(x, np.array([0, 1]))
        self.assertEqual(names[0], "event_time_normalized")
        self.assertAlmostEqual(float(features[2, 0]), 2.0)

    def test_edge_order_proxy_fails_permutation_invariance(self):
        edges = np.array([[0, 1], [0, 2], [2, 1], [1, 2], [2, 0]])
        self.assertGreater(edge_order_permutation_change(edges, 3, seed=7), 0.0)


if __name__ == "__main__":
    unittest.main()
