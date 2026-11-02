"""
RGTS Pipeline Module
High-level orchestration for Reference-Guided Test Subsampling and behavioral evaluation.
"""

from typing import List, Dict, Any, Optional

try:
    from .clustering import cluster_solutions_subset, get_ari, calculate_similarity_subset
    from .rgts_selectors import (
        select_m1_rgts, select_b0_full, select_b1_random,
        select_b2_input_stratified, select_b3_input_diversity, select_b4_output_stratified
    )
    from .feature_encoder import extract_input_features, extract_reference_strata
except (ImportError, ValueError):
    from clustering import cluster_solutions_subset, get_ari, calculate_similarity_subset
    from rgts_selectors import (
        select_m1_rgts, select_b0_full, select_b1_random,
        select_b2_input_stratified, select_b3_input_diversity, select_b4_output_stratified
    )
    from feature_encoder import extract_input_features, extract_reference_strata


class RGTSPipeline:
    """
    High-level orchestration interface for RGTS test selection and behavioral clustering.
    """
    SELECTORS = {
        "B0": select_b0_full,
        "B1": select_b1_random,
        "B2": select_b2_input_stratified,
        "B3": select_b3_input_diversity,
        "B4": select_b4_output_stratified,
        "M1": select_m1_rgts,
        "RGTS": select_m1_rgts,
    }

    def __init__(self, method: str = "M1", budget: int = 32, seed: int = 42, threshold: float = 0.99):
        self.method = method.upper()
        self.budget = budget
        self.seed = seed
        self.threshold = threshold
        if self.method not in self.SELECTORS:
            raise ValueError(f"Unknown selector method: {method}. Choose from {list(self.SELECTORS.keys())}")

    def select_tests(self, inputs: List[str], ref_outputs: List[str]) -> List[int]:
        """Select indices of tests using the configured method."""
        selector = self.SELECTORS[self.method]
        return selector(inputs, ref_outputs, k=self.budget, seed=self.seed)

    def evaluate_cohort(
        self,
        inputs: List[str],
        ref_outputs: List[str],
        candidate_behaviors: Dict[str, Dict[str, str]],
        ref_name: str = "reference_solution"
    ) -> Dict[str, Any]:
        """
        Runs test selection and clusters candidate behaviors on the subsampled test suite.
        """
        selected_indices = self.select_tests(inputs, ref_outputs)
        selected_keys = [str(inputs[i]) for i in selected_indices]

        # Build reference behavior dictionary using actual input text as keys
        ref_beh = {str(inputs[i]): str(ref_outputs[i]) for i in range(len(inputs))}
        all_sols = {ref_name: ref_beh}
        all_sols.update(candidate_behaviors)

        clusters = cluster_solutions_subset(
            all_sols,
            selected_keys,
            similarity_threshold=self.threshold,
            use_common_keys=False
        )

        return {
            "method": self.method,
            "budget": self.budget,
            "selected_indices": selected_indices,
            "selected_keys": selected_keys,
            "clusters": clusters,
            "n_clusters": len(clusters),
        }

