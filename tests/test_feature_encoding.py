"""
Test Feature Encoding and Semantic Strata Classification
Verifies candidate-blind feature extraction dimensions, normalization properties,
and reference output classification.
"""

import os
import sys
import unittest
import numpy as np

cur_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.abspath(os.path.join(cur_dir, "..", "src"))
for d in [cur_dir, src_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

from feature_encoder import extract_input_features, extract_reference_strata


class TestFeatureEncoding(unittest.TestCase):
    def test_feature_dimensions_and_types(self):
        sample_inputs = [
            "0",
            "-15",
            "[1, 2, 3]",
            "[[1, 2], [3, 4]]",
            "hello world test",
            "1000000000",
            ""
        ]
        feats = extract_input_features(sample_inputs)
        self.assertEqual(feats.shape, (len(sample_inputs), 7))
        self.assertEqual(feats.dtype, np.float64)

    def test_constant_column_normalization_no_nan(self):
        # All inputs have same length and no negatives
        identical_inputs = ["abc", "def", "ghi"]
        feats = extract_input_features(identical_inputs)
        self.assertFalse(np.isnan(feats).any(), "NaN detected in feature matrix!")
        self.assertFalse(np.isinf(feats).any(), "Inf detected in feature matrix!")

    def test_reference_strata_classification(self):
        cases = [
            ("", 0),
            ("none", 0),
            ("false", 0),
            ("0", 0),
            ("[]", 0),
            ("true", 1),
            ("1", 1),
            ("42", 2),
            ("-3.14", 2),
            ("[1, 2, 3]", 3),
            ("[1, 2, 3, 4, 5, 6, 7]", 4),
            ("custom_string_object", 5)
        ]
        outputs = [c[0] for c in cases]
        expected_strata = [c[1] for c in cases]
        actual_strata = extract_reference_strata(outputs)
        self.assertEqual(actual_strata, expected_strata)


if __name__ == "__main__":
    unittest.main()

