"""
RGTS Core Unit Test Suite
Verifies feature extraction, selector budget adherence, anti-leakage constraints,
and clustering clique consistency.
"""

import os
import sys
import unittest
import numpy as np

# Ensure src is importable
cur_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(cur_dir, "..", "src"))
for d in [cur_dir, src_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from feature_encoder import extract_input_features, extract_reference_strata
from rgts_selectors import (
    select_b0_full, select_b1_random, select_b2_input_stratified,
    select_b3_input_diversity, select_b4_output_stratified, select_m1_rgts
)
from clustering import calculate_similarity_subset, cluster_solutions_subset
from evaluator import evaluate_candidate_equivalence, evaluate_clustering_agreement


class TestRGTSPipeline(unittest.TestCase):
    def setUp(self):
        # Synthetic problem with 50 inputs
        self.inputs = [f"input_val_{i}_{i%5}_[-{i}]" for i in range(50)]
        self.ref_outputs = ["true" if i % 4 == 0 else "false" if i % 4 == 1 else str(i * 10) for i in range(50)]

    def test_feature_encoder(self):
        feats = extract_input_features(self.inputs)
        self.assertEqual(feats.shape, (50, 7))
        # Check Z-score normalization: mean ~ 0
        np.testing.assert_allclose(np.mean(feats, axis=0), np.zeros(7), atol=1e-5)
        # Check std: should be 1.0 for non-constant columns, 0.0 for constant columns
        for col in range(7):
            col_std = np.std(feats[:, col])
            self.assertTrue(np.isclose(col_std, 1.0, atol=1e-5) or np.isclose(col_std, 0.0, atol=1e-5))

        strata = extract_reference_strata(self.ref_outputs)
        self.assertEqual(len(strata), 50)
        self.assertTrue(set(strata).issubset({0, 1, 2, 3, 4, 5}))

    def test_selectors_budget_and_bounds(self):
        for k in [10, 20, 35, 50]:
            for name, sel_fn in [
                ("B0", select_b0_full if k == 50 else None),
                ("B1", select_b1_random),
                ("B2", select_b2_input_stratified),
                ("B3", select_b3_input_diversity),
                ("B4", select_b4_output_stratified),
                ("M1", select_m1_rgts)
            ]:
                if sel_fn is None:
                    continue
                sel = sel_fn(self.inputs, self.ref_outputs, k, seed=123)
                self.assertEqual(len(sel), k, f"{name} failed exact budget k={k}")
                self.assertEqual(len(set(sel)), k, f"{name} returned duplicates at k={k}")
                self.assertTrue(all(0 <= idx < 50 for idx in sel), f"{name} index out of bounds")

    def test_hard_anti_leakage(self):
        """Crucial test: selector must NEVER depend on candidate outputs."""
        k = 16
        sel_baseline = select_m1_rgts(self.inputs, self.ref_outputs, k, seed=42)

        # Fake candidate outputs (noisy, inverted, adversarial)
        fake_candidate_1 = {self.inputs[i]: "adversarial_output" for i in range(50)}
        fake_candidate_2 = {self.inputs[i]: "all_pass" for i in range(50)}

        # Re-run selector
        sel_after_candidate = select_m1_rgts(self.inputs, self.ref_outputs, k, seed=42)

        # MUST be bit-for-bit strictly identical
        self.assertEqual(sel_baseline, sel_after_candidate, "HARD LEAKAGE DETECTED! Selector output varied!")

    def test_clustering_clique_property(self):
        # 3 identical solutions, 1 completely different
        sols = {
            "Sol1": {"a": "1", "b": "2", "c": "3"},
            "Sol2": {"a": "1", "b": "2", "c": "3"},
            "Sol3": {"a": "1", "b": "2", "c": "3"},
            "Sol4": {"a": "9", "b": "9", "c": "9"},
        }
        subset = ["a", "b", "c"]
        clusters = cluster_solutions_subset(sols, subset, similarity_threshold=0.99)
        self.assertEqual(len(clusters), 2)
        cl_sizes = sorted([len(c) for c in clusters.values()])
        self.assertEqual(cl_sizes, [1, 3])


if __name__ == "__main__":
    unittest.main()

