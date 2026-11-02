"""
RGTS Feature Encoder Module
Extracts candidate-blind input features (7 structural dimensions) and
semantic reference output strata for stratified test subsampling.
"""

import math
import re
from typing import List
import numpy as np


def extract_input_features(inputs: List[str]) -> np.ndarray:
    """
    Candidate-blind feature extraction from raw input strings.
    Features:
    0: char_length (log1p)
    1: token_count (log1p)
    2: has_negative (0 or 1)
    3: has_zero (0 or 1)
    4: max_numeric_magnitude (log1p)
    5: is_nested (0 or 1)
    6: char_entropy
    """
    matrix = []
    for inp in inputs:
        s = str(inp).strip()
        length = len(s)
        tokens = re.split(r'[\s,\[\]\(\)]+', s)
        tokens = [t for t in tokens if t]
        token_count = len(tokens)

        has_neg = 1.0 if ('-' in s) else 0.0
        has_zero = 1.0 if re.search(r'\b0\b', s) else 0.0

        # Max numeric magnitude
        nums = [abs(float(t)) for t in tokens if re.match(r'^-?\d+(?:\.\d+)?$', t)]
        max_num = max(nums) if nums else 0.0

        # Nested brackets
        is_nested = 1.0 if ('[[' in s or '([' in s) else 0.0

        # Char entropy
        char_counts = {}
        for c in s:
            char_counts[c] = char_counts.get(c, 0) + 1
        entropy = 0.0
        if length > 0:
            for count in char_counts.values():
                p = count / length
                entropy -= p * math.log2(p)

        matrix.append([
            math.log1p(length),
            math.log1p(token_count),
            has_neg,
            has_zero,
            math.log1p(max_num),
            is_nested,
            entropy
        ])

    mat = np.array(matrix, dtype=float)
    # Z-score normalize per feature column with numerical threshold
    mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0)
    zero_std_mask = std < 1e-8
    std[zero_std_mask] = 1.0
    normalized = (mat - mean) / std
    normalized[:, zero_std_mask] = 0.0
    return normalized


def extract_reference_strata(ref_outputs: List[str]) -> List[int]:
    """
    Candidate-blind semantic classification of reference outputs.
    Classes:
    0: Empty / None / 'false' / '0'
    1: Boolean True / 'true' / '1'
    2: Single integer / float
    3: Short sequence (length <= 5 or simple list)
    4: Long sequence / complex structure
    5: Other / Unclassified
    """
    classes = []
    for out in ref_outputs:
        s = str(out).strip().lower()
        if s in ('', 'none', 'null', 'false', '0', '[]', '{}'):
            c = 0
        elif s in ('true', '1'):
            c = 1
        elif re.match(r'^-?\d+(?:\.\d+)?$', s):
            c = 2
        elif s.startswith('[') or s.startswith('{') or ',' in s:
            elements = s.count(',') + 1
            if elements <= 5:
                c = 3
            else:
                c = 4
        else:
            c = 5
        classes.append(c)
    return classes

