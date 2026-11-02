"""
RGTS Clustering Module
Provides representative-based clique clustering, pairwise similarity calculation,
and Adjusted Rand Index (ARI) computation with strict handling of missing values.
"""

from itertools import combinations
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import adjusted_rand_score


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1

    def make_set(self, item):
        self.parent[item] = item
        self.rank[item] = 0


def calculate_similarity_subset(
    sol_a: Dict[str, str],
    sol_b: Dict[str, str],
    subset_keys: List[str],
    use_common_keys: bool = False
) -> float:
    """
    Calculates behavioral similarity restricted to a specific test subset.

    Parameters
    ----------
    sol_a : dict
        Execution behavior dictionary of candidate A.
    sol_b : dict
        Execution behavior dictionary of candidate B.
    subset_keys : list
        The subset of test input keys to evaluate.
    use_common_keys : bool, default=False
        If True, calculates matching / len(common_keys) on common tests executed by both
        candidates (matching Eagal et al. get_clusters.py lines 128-135).
        If False (default), calculates matches / len(subset_keys) with fixed budget denominator,
        where missing keys (None) are strictly never treated as matches.
    """
    if not subset_keys:
        return 1.0 if sol_a == {} and sol_b == {} else 0.0

    # If either had an error sentinel, they cannot match valid tests
    if sol_a.get("__ERROR__") or sol_b.get("__ERROR__"):
        return 1.0 if (sol_a.get("__ERROR__") and sol_b.get("__ERROR__")) else 0.0

    if use_common_keys:
        common = set(sol_a.keys()) & set(sol_b.keys()) & set(subset_keys)
        if not common:
            return 0.0
        matching_pairs = sum(1 for k in common if sol_a[k] == sol_b[k])
        return matching_pairs / len(common)
    else:
        # Strict fixed denominator: missing outputs (None) are NEVER counted as matches
        matches = sum(
            1 for k in subset_keys
            if sol_a.get(k) is not None and sol_b.get(k) is not None and sol_a[k] == sol_b[k]
        )
        return matches / len(subset_keys)


def cluster_solutions_subset(
    solutions_behavior: Dict[str, Dict[str, str]],
    subset_keys: List[str],
    similarity_threshold: float = 0.99,
    use_common_keys: bool = False
) -> Dict[int, List[str]]:
    """
    Representative-based clustering algorithm on a test subset:
    1. Union-Find on pairwise similarities meeting threshold.
    2. Clique refinement to guarantee all members within each cluster are pairwise >= threshold.
    """
    uf = UnionFind()
    for sol in solutions_behavior.keys():
        uf.make_set(sol)

    # Initial union-find
    for (sol_a, data_a), (sol_b, data_b) in combinations(solutions_behavior.items(), 2):
        sim = calculate_similarity_subset(data_a, data_b, subset_keys, use_common_keys=use_common_keys)
        if sim >= similarity_threshold:
            uf.union(sol_a, sol_b)

    temp_clusters = {}
    for sol in solutions_behavior.keys():
        root = uf.find(sol)
        temp_clusters.setdefault(root, []).append(sol)

    # Refine clusters: intra-cluster pairwise verification
    final_clusters = []
    for cluster in temp_clusters.values():
        new_clusters = []
        for sol in cluster:
            placed = False
            for c in new_clusters:
                if all(
                    calculate_similarity_subset(
                        solutions_behavior[sol], solutions_behavior[other],
                        subset_keys, use_common_keys=use_common_keys
                    ) >= similarity_threshold
                    for other in c
                ):
                    c.append(sol)
                    placed = True
                    break
            if not placed:
                new_clusters.append([sol])
        final_clusters.extend(new_clusters)

    return {i + 1: c for i, c in enumerate(final_clusters)}


def cluster_solutions(
    solutions_behavior: Dict[str, Dict[str, str]],
    similarity_threshold: float = 0.99
) -> Dict[int, List[str]]:
    """
    Baseline clustering across all available common keys (equivalent to Eagal et al.).
    """
    all_keys = set()
    for beh in solutions_behavior.values():
        all_keys.update(beh.keys())
    return cluster_solutions_subset(
        solutions_behavior,
        list(all_keys),
        similarity_threshold=similarity_threshold,
        use_common_keys=True
    )


def get_ari(
    clusters_gold: Dict[int, List[str]],
    clusters_sub: Dict[int, List[str]],
    sol_order: List[str]
) -> float:
    """
    Computes Adjusted Rand Index between two clustering partitions over sol_order.
    """
    if len(sol_order) <= 1:
        return 1.0
    g_map = {m: cid for cid, members in clusters_gold.items() for m in members}
    s_map = {m: cid for cid, members in clusters_sub.items() for m in members}
    lg = [g_map.get(s, -1) for s in sol_order]
    ls = [s_map.get(s, -2) for s in sol_order]
    return float(adjusted_rand_score(lg, ls))

