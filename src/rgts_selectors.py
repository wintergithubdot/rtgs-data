"""
RGTS Test Subsampling Selectors Module
Implements baselines B0 (Full), B1 (Random), B2 (Input Stratified),
B3 (Input Diversity Farthest-First), B4 (Output Stratified), and
M1 (RGTS Reference-Guided Test Subsampling).
All selectors are strictly candidate-blind (never access candidate outputs).
"""

import random
from typing import List
import numpy as np

try:
    from .feature_encoder import extract_input_features, extract_reference_strata
except (ImportError, ValueError):
    from feature_encoder import extract_input_features, extract_reference_strata


def select_b0_full(inputs: List[str], ref_outputs: List[str], k: int = 256, seed: int = 42) -> List[int]:
    """Baseline B0: Full test suite (all N inputs)."""
    return list(range(len(inputs)))


def select_b1_random(inputs: List[str], ref_outputs: List[str], k: int, seed: int = 42) -> List[int]:
    """Baseline B1: Uniform random sampling without replacement."""
    assert k <= len(inputs), f"k ({k}) exceeds inputs length ({len(inputs)})"
    rng = random.Random(seed)
    return sorted(rng.sample(range(len(inputs)), k))


def select_b2_input_stratified(inputs: List[str], ref_outputs: List[str], k: int, seed: int = 42) -> List[int]:
    """Baseline B2: Input length/token stratified random sampling."""
    assert k <= len(inputs)
    lengths = [len(str(x)) for x in inputs]
    # 4 quantile bins
    quantiles = np.quantile(lengths, [0.25, 0.5, 0.75])
    strata = {0: [], 1: [], 2: [], 3: []}
    for idx, l in enumerate(lengths):
        if l <= quantiles[0]:
            strata[0].append(idx)
        elif l <= quantiles[1]:
            strata[1].append(idx)
        elif l <= quantiles[2]:
            strata[2].append(idx)
        else:
            strata[3].append(idx)

    rng = random.Random(seed)
    selected = []
    # Distribute quota
    active_strata = [s for s in strata.values() if s]
    base_quota = k // len(active_strata)
    rem = k % len(active_strata)

    for i, s in enumerate(active_strata):
        q = base_quota + (1 if i < rem else 0)
        q = min(q, len(s))
        selected.extend(rng.sample(s, q))

    # Fill any remaining slots
    if len(selected) < k:
        rem_pool = [i for i in range(len(inputs)) if i not in selected]
        selected.extend(rng.sample(rem_pool, k - len(selected)))

    return sorted(selected[:k])


def select_b3_input_diversity(inputs: List[str], ref_outputs: List[str], k: int, seed: int = 42) -> List[int]:
    """Baseline B3: Input-only farthest-first diversity traversal (no output guidance)."""
    assert k <= len(inputs)
    if k == len(inputs):
        return list(range(len(inputs)))

    feats = extract_input_features(inputs)
    rng = random.Random(seed)

    # First item: medoid or deterministic seed choice
    start_idx = rng.randint(0, len(inputs) - 1)
    selected = [start_idx]

    # Precompute pairwise distances to selected set
    min_dists = np.linalg.norm(feats - feats[start_idx], axis=1)

    while len(selected) < k:
        # Mask already selected
        min_dists[selected] = -1.0
        # Pick point with maximum minimum distance
        best_idx = int(np.argmax(min_dists))
        selected.append(best_idx)

        # Update min distances with new point
        new_dists = np.linalg.norm(feats - feats[best_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return sorted(selected)


def select_b4_output_stratified(inputs: List[str], ref_outputs: List[str], k: int, seed: int = 42) -> List[int]:
    """Baseline B4: Reference-output stratified random sampling."""
    assert k <= len(inputs)
    strata_labels = extract_reference_strata(ref_outputs)
    strata = {}
    for idx, s in enumerate(strata_labels):
        strata.setdefault(s, []).append(idx)

    rng = random.Random(seed)
    selected = []

    # Proportional allocation
    total_n = len(inputs)
    quotas = {}
    alloc_sum = 0
    for s_cls, members in strata.items():
        q = max(1, int(round(k * len(members) / total_n)))
        q = min(q, len(members))
        quotas[s_cls] = q
        alloc_sum += q

    # Adjust to exact k
    while alloc_sum > k:
        eligible = [s for s in quotas if quotas[s] > 1]
        max_s = max(eligible, key=lambda s: (quotas[s], -s))
        quotas[max_s] -= 1
        alloc_sum -= 1
    while alloc_sum < k:
        cand_s = [s for s in strata.keys() if quotas[s] < len(strata[s])]
        if not cand_s:
            break
        best_s = max(cand_s, key=lambda s: len(strata[s]) - quotas[s])
        quotas[best_s] += 1
        alloc_sum += 1

    for s_cls, q in quotas.items():
        selected.extend(rng.sample(strata[s_cls], q))

    # Fallback if needed
    if len(selected) < k:
        rem_pool = [i for i in range(len(inputs)) if i not in selected]
        selected.extend(rng.sample(rem_pool, k - len(selected)))

    return sorted(selected[:k])


def select_m1_rgts(inputs: List[str], ref_outputs: List[str], k: int, seed: int = 42) -> List[int]:
    """
    Proposed Method M1 (RGTS):
    Reference-Guided Test Subsampling combining Reference Output Stratification
    with Farthest-First Input Diversity within each semantic stratum.
    Candidate-blind: strictly never accesses candidate outputs.
    """
    assert k <= len(inputs)
    if k == len(inputs):
        return list(range(len(inputs)))

    feats = extract_input_features(inputs)
    strata_labels = extract_reference_strata(ref_outputs)
    strata = {}
    for idx, s in enumerate(strata_labels):
        strata.setdefault(s, []).append(idx)

    total_n = len(inputs)
    # Proportional quotas
    quotas = {}
    alloc_sum = 0
    for s_cls, members in strata.items():
        q = max(1, int(round(k * len(members) / total_n)))
        q = min(q, len(members))
        quotas[s_cls] = q
        alloc_sum += q

    while alloc_sum > k:
        eligible = [s for s in quotas if quotas[s] > 1]
        max_s = max(eligible, key=lambda s: (quotas[s], -s))
        quotas[max_s] -= 1
        alloc_sum -= 1
    while alloc_sum < k:
        cand_s = [s for s in strata.keys() if quotas[s] < len(strata[s])]
        if not cand_s:
            break
        best_s = max(cand_s, key=lambda s: (len(strata[s]) - quotas[s], -s))
        quotas[best_s] += 1
        alloc_sum += 1

    rng = random.Random(seed)
    selected = []

    # Farthest-first traversal within each stratum
    for s_cls, q in quotas.items():
        members = strata[s_cls]
        if q == len(members):
            selected.extend(members)
            continue

        # First point: deterministic medoid or seeded item
        strat_feats = feats[members]
        centroid = np.mean(strat_feats, axis=0)
        # Point furthest from centroid to capture boundary conditions
        first_local_idx = int(np.argmax(np.linalg.norm(strat_feats - centroid, axis=1)))
        stratum_selected = [members[first_local_idx]]

        # Min distances to selected within this stratum
        min_dists = np.linalg.norm(strat_feats - strat_feats[first_local_idx], axis=1)

        while len(stratum_selected) < q:
            # Mask selected
            local_sel_indices = [members.index(m) for m in stratum_selected]
            min_dists[local_sel_indices] = -1.0
            best_local_idx = int(np.argmax(min_dists))
            stratum_selected.append(members[best_local_idx])

            new_dists = np.linalg.norm(strat_feats - strat_feats[best_local_idx], axis=1)
            min_dists = np.minimum(min_dists, new_dists)

        selected.extend(stratum_selected)

    if len(selected) < k:
        rem_pool = [i for i in range(len(inputs)) if i not in selected]
        selected.extend(rng.sample(rem_pool, k - len(selected)))

    return sorted(selected[:k])

