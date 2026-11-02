"""
RGTS: Reference-Guided Test Subsampling
Cost-effective evaluation of LLM-generated code behavioral equivalence.
"""

from .clustering import (
    calculate_similarity_subset,
    cluster_solutions_subset,
    cluster_solutions,
    get_ari,
    UnionFind
)
from .feature_encoder import (
    extract_input_features,
    extract_reference_strata
)
from .rgts_selectors import (
    select_b0_full,
    select_b1_random,
    select_b2_input_stratified,
    select_b3_input_diversity,
    select_b4_output_stratified,
    select_m1_rgts
)
from .evaluator import (
    evaluate_candidate_equivalence,
    evaluate_clustering_agreement
)
from .pipeline import RGTSPipeline

__version__ = "1.0.0"
__all__ = [
    "calculate_similarity_subset",
    "cluster_solutions_subset",
    "cluster_solutions",
    "get_ari",
    "UnionFind",
    "extract_input_features",
    "extract_reference_strata",
    "select_b0_full",
    "select_b1_random",
    "select_b2_input_stratified",
    "select_b3_input_diversity",
    "select_b4_output_stratified",
    "select_m1_rgts",
    "evaluate_candidate_equivalence",
    "evaluate_clustering_agreement",
    "RGTSPipeline",
]

