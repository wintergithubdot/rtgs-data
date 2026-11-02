"""
Test Level 1: Synthetic Transitive Chain and Intra-Cluster Verification
Verifies that RGTS clustering strictly matches baseline behavior when handling
transitive relationships where A~B and B~C, but A!~C.
"""

import os
import sys
import unittest

cur_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(cur_dir, "..", "src"))
for d in [cur_dir, src_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from clustering import cluster_solutions_subset, calculate_similarity_subset, cluster_solutions


class TestSyntheticEquivalence(unittest.TestCase):
    def test_transitive_non_compatible_separation(self):
        keys = [f"k_{i}" for i in range(100)]
        # SolA: all "0"
        sol_a = {k: "0" for k in keys}
        # SolB: 1 difference (99% match with A)
        sol_b = {k: ("1" if i == 0 else "0") for i, k in enumerate(keys)}
        # SolC: 2 differences (99% match with B, 98% match with A)
        sol_c = {k: ("1" if i in (0, 1) else "0") for i, k in enumerate(keys)}

        beh = {"SolA": sol_a, "SolB": sol_b, "SolC": sol_c}

        # Baseline clustering (common keys)
        c_baseline = cluster_solutions(beh, similarity_threshold=0.99)
        # RGTS clustering (subset keys)
        c_rgts = cluster_solutions_subset(beh, keys, similarity_threshold=0.99, use_common_keys=False)

        # Both must produce identical partitions: {1: ['SolA', 'SolB'], 2: ['SolC']}
        self.assertEqual(c_baseline, c_rgts)
        self.assertEqual(len(c_rgts), 2)
        # Check that SolA and SolB are together, and SolC is separated
        c1 = next(c for c in c_rgts.values() if "SolA" in c)
        self.assertIn("SolB", c1)
        self.assertNotIn("SolC", c1)


if __name__ == "__main__":
    unittest.main()

