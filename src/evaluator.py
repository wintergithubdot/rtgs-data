"""
RGTS Evaluator Module
Computes behavioral equivalence and clustering agreement metrics (ARI, clique consistency).
"""

from typing import Dict, List, Tuple
from sklearn.metrics import adjusted_rand_score

try:
    from .clustering import calculate_similarity_subset, cluster_solutions_subset
except (ImportError, ValueError):
    from clustering import calculate_similarity_subset, cluster_solutions_subset


def evaluate_candidate_equivalence(
    ref_behavior: Dict[str, str],
    candidate_behaviors: Dict[str, Dict[str, str]],
    subset_keys: List[str],
    threshold: float = 0.99
) -> Dict[str, int]:
    """
    Evaluates whether each candidate is behaviorally equivalent to reference (1) or not (0)
    on the chosen subset_keys.
    """
    labels = {}
    for cand_name, cand_data in candidate_behaviors.items():
        sim = calculate_similarity_subset(cand_data, ref_behavior, subset_keys)
        labels[cand_name] = 1 if sim >= threshold else 0
    return labels


def evaluate_clustering_agreement(
    ref_name: str,
    candidate_behaviors: Dict[str, Dict[str, str]],
    ref_behavior: Dict[str, str],
    subset_keys: List[str],
    gold_clusters: Dict[int, List[str]],
    threshold: float = 0.99
) -> Tuple[float, int, int]:
    """
    Runs clustering on subset_keys and computes:
    1. Adjusted Rand Index (ARI) compared to gold_clusters
    2. Number of candidates in the cluster containing the reference solution
    3. Agreement on reference cluster membership (binary 1/0)
    """
    all_sols = {ref_name: ref_behavior}
    all_sols.update(candidate_behaviors)

    sub_clusters = cluster_solutions_subset(all_sols, subset_keys, threshold)

    sol_order = sorted(all_sols.keys())
    gold_map = {m: cid for cid, members in gold_clusters.items() for m in members}
    sub_map = {m: cid for cid, members in sub_clusters.items() for m in members}

    labels_gold = [gold_map.get(s, -1) for s in sol_order]
    labels_sub = [sub_map.get(s, -2) for s in sol_order]

    ari = float(adjusted_rand_score(labels_gold, labels_sub))

    ref_cluster_gold = set()
    for members in gold_clusters.values():
        if ref_name in members:
            ref_cluster_gold = set(members) - {ref_name}
            break

    ref_cluster_sub = set()
    for members in sub_clusters.values():
        if ref_name in members:
            ref_cluster_sub = set(members) - {ref_name}
            break

    ref_cluster_match = 1 if ref_cluster_gold == ref_cluster_sub else 0
    sub_ref_count = len(ref_cluster_sub)

    return ari, sub_ref_count, ref_cluster_match

