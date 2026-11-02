"""
RGTS Pipeline Integration Test
Verifies end-to-end integration of RGTSPipeline:
- Text-based input keys representation
- Equivalence clustering between identical candidates and reference
- Divergent candidates separation under similarity threshold
- Prevention of false matches from missing keys
- Determinism on repeated invocations
"""

import os
import sys
import unittest

# Ensure src is importable
cur_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(cur_dir, "..", "src"))
for d in [cur_dir, src_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from pipeline import RGTSPipeline


class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        # Text-based input keys matching actual benchmark trace format
        self.inputs = ["[1,2,3]", "[-5,0,8]", "[10,20]", "[0,0,0]", "[-1,-2]"]
        self.ref_outputs = ["6", "3", "30", "0", "-3"]

        self.cand_identical = {
            "[1,2,3]": "6",
            "[-5,0,8]": "3",
            "[10,20]": "30",
            "[0,0,0]": "0",
            "[-1,-2]": "-3",
        }

        self.cand_divergent = {
            "[1,2,3]": "999",
            "[-5,0,8]": "999",
            "[10,20]": "999",
            "[0,0,0]": "999",
            "[-1,-2]": "999",
        }

        # Candidate with missing keys (only executes first test)
        self.cand_partial_missing = {
            "[1,2,3]": "6",
        }

        # Candidates with completely missing execution traces
        self.cand_empty_a = {}
        self.cand_empty_b = {}

    def test_pipeline_cohort_evaluation_key_fidelity(self):
        pipeline = RGTSPipeline(method="M1", budget=3, seed=42, threshold=0.99)
        res = pipeline.evaluate_cohort(
            inputs=self.inputs,
            ref_outputs=self.ref_outputs,
            candidate_behaviors={
                "cand_id": self.cand_identical,
                "cand_div": self.cand_divergent,
            },
            ref_name="ref"
        )

        # 1. Budget and index checks
        self.assertEqual(len(res["selected_indices"]), 3)
        self.assertEqual(len(res["selected_keys"]), 3)

        # 2. Verify selected_keys are actual input strings, NOT stringified indices
        for k in res["selected_keys"]:
            self.assertIn(k, self.inputs)
            self.assertFalse(k.isdigit(), f"Selected key '{k}' must not be a bare digit index!")

        for idx, k in zip(res["selected_indices"], res["selected_keys"]):
            self.assertEqual(self.inputs[idx], k)

        # 3. Verify clustering: identical candidate MUST cluster with reference
        clusters = res["clusters"]
        ref_cluster = None
        for cid, members in clusters.items():
            if "ref" in members:
                ref_cluster = members
                break

        self.assertIsNotNone(ref_cluster, "Reference solution must be in a cluster")
        self.assertIn("cand_id", ref_cluster, "Candidate with identical behavior must be in reference cluster")
        self.assertNotIn("cand_div", ref_cluster, "Candidate with divergent behavior must NOT be in reference cluster")

    def test_pipeline_missing_keys_no_false_matches(self):
        pipeline = RGTSPipeline(method="M1", budget=3, seed=42, threshold=0.99)
        res = pipeline.evaluate_cohort(
            inputs=self.inputs,
            ref_outputs=self.ref_outputs,
            candidate_behaviors={
                "cand_id": self.cand_identical,
                "cand_part": self.cand_partial_missing,
                "cand_empty_a": self.cand_empty_a,
                "cand_empty_b": self.cand_empty_b,
            },
            ref_name="ref"
        )

        clusters = res["clusters"]
        ref_cluster = None
        for cid, members in clusters.items():
            if "ref" in members:
                ref_cluster = members
                break

        # Partial missing candidate only has 1 match at most, similarity <= 1/3 < 0.99
        self.assertNotIn("cand_part", ref_cluster, "Partial missing candidate must NOT falsely match reference")

        # Two empty candidates must not cluster together into a false match
        for cid, members in clusters.items():
            if "cand_empty_a" in members:
                self.assertNotIn("cand_empty_b", members, "Two empty candidates must not produce false positive cluster")

    def test_pipeline_determinism(self):
        pipeline1 = RGTSPipeline(method="M1", budget=3, seed=12345, threshold=0.99)
        pipeline2 = RGTSPipeline(method="M1", budget=3, seed=12345, threshold=0.99)

        cands = {
            "cand_id": self.cand_identical,
            "cand_div": self.cand_divergent,
        }

        res1 = pipeline1.evaluate_cohort(self.inputs, self.ref_outputs, cands, "ref")
        res2 = pipeline2.evaluate_cohort(self.inputs, self.ref_outputs, cands, "ref")

        self.assertEqual(res1["selected_indices"], res2["selected_indices"])
        self.assertEqual(res1["selected_keys"], res2["selected_keys"])
        self.assertEqual(res1["clusters"], res2["clusters"])


if __name__ == "__main__":
    unittest.main()

