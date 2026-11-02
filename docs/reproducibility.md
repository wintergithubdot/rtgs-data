# Reproduction Guide: RGTS Research Artifacts

This guide provides reproducible verification procedures for the empirical results published in our manuscript:
> **"Reference-Guided Test Subsampling for Behavioral Verification of Large Language Model Generated Code Clones"**  
> *Target Journal: The Journal of Systems and Software (Elsevier)*

---

## Three-Level Verification Hierarchy

```mermaid
flowchart TD
    L1["Level 1: Synthetic Transitive Refinement<br/>(Execution time: ~0.1s)"]
    L2["Level 2: Real Benchmark Cohort Fixtures<br/>(Execution time: ~1.5s)"]
    L3A["Level 3A: Standalone Dynamic Ledger Census<br/>(Execution time: ~2.5s)"]
    L3B["Level 3B: Full Live Recomputation from data.zip<br/>(Clustering time: 1,114.7s / 18m 35s; Total: 1,124.6s / 18m 44s)"]

    L1 --> L2
    L2 --> L3A
    L3A -.->|Requires data.zip (standalone from Zenodo)| L3B
```

### Level 1: Synthetic Transitive Refinement (`test_1_synthetic_transitive_refinement`)
- **Purpose**: Mathematically proves that RGTS's Union-Find and intra-cluster pairwise verification algorithm correctly separates transitive non-compatible candidate solutions (where $A \sim B \ge 0.99$ and $B \sim C \ge 0.99$, but $A \sim C < 0.99$).
- **Runtime**: Instantaneous (< 0.1 second).
- **Data Requirement**: None (pure in-memory synthetic fixture).

### Level 2: Real Benchmark Cohort Fixtures (`test_2_real_benchmark_cohorts_fixture`)
- **Purpose**: Evaluates live clustering partitions on 5 real benchmark cohorts extracted from the original study across diverse conditions (easy/medium/hard, temperatures 0, 0.5, 1.0, multiple models and translation tasks), including a known partial-output cohort.
- **Runtime**: ~1.5 seconds.
- **Data Requirement**: `fixtures/real_benchmark_cohorts_fixture.json` (1.6 MB, included in repository).

### Level 3A: Computational Census and Cohort Ledger Verification (`test_3_computational_census_and_cohort_audit`)
- **Purpose**: Dynamically parses the candidate census ledger (25,920 rows) and cohort verification ledger (1,296 rows), mathematically recalculating summary statistics, partition equivalence rates (1,292/1,296 = 99.69% on fixed denominator $|T_k|$ and 1,296/1,296 = 100.0% on common keys), and sensitivity bounds.
- **Runtime**: ~2.5 seconds.
- **Data Requirement**: `results/audit_census_25920_candidates.csv` and `results/audit_clustering_1296_cohorts.csv` (included in repository).

### Level 3B: Full Live Recomputation (`--full-1296`)
- **Purpose**: Extracts every candidate output file from the raw `data.zip` archive published by Eagal et al. (Zenodo DOI: [10.5281/zenodo.14968618](https://doi.org/10.5281/zenodo.14968618)), performing live end-to-end clustering across all 1,296 cohorts (36 problems $\times$ 3 temperatures $\times$ 4 models $\times$ 3 tasks) and matching every result 1-to-1 against the ledger.
- **Runtime**: Approximately 1,114.7 seconds (18 minutes 35 seconds) for raw trace extraction and clustering recomputation; total script execution duration: 1,124.6 seconds (18 minutes 44 seconds). Note: This duration is an empirical measurement conducted on our specific experimental environment (Intel Core i7, 32 GB RAM, NVMe SSD on Windows 64-bit) and is provided for benchmark transparency rather than as a uniform hardware performance guarantee across all systems.
- **Data Requirement**: `data.zip` (standalone download from Zenodo; pass via `--data-zip <path>`). If omitted, the script safely stops with an informative error message and exit code 1.
- **Data Provenance**: The raw benchmark dataset (`data.zip`) and baseline evaluation code are third-party artifacts published by Eagal et al. (2025). The RGTS repository contains our independent RGTS method implementation, verification harnesses, replication fixtures, and empirical ledgers.

---

## Quickstart Commands

```bash
# 1. Enter the repository root and install dependencies
cd RGTS
pip install -r requirements.txt

# 2. Run unit and integration tests (11/11 tests pass in < 2s)
pytest tests/ -v
# or:
python -m unittest discover tests

# 3. Run Level 1, 2, and 3A Verification (instant, no raw data.zip download required)
python scripts/verify_clustering_equivalence.py

# 4. Reproduce empirical tables and dynamic Wilcoxon / exact permutation tests
python scripts/reproduce_results.py

# 5. (Optional) Run Full Live 1,296-Cohort Recomputation from raw data.zip
# Obtain standalone data.zip from Zenodo DOI: 10.5281/zenodo.14968618
python scripts/verify_clustering_equivalence.py --full-1296 --data-zip /path/to/data.zip
```
