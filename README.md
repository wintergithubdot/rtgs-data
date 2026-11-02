# Reference-Guided Test Subsampling for Behavioral Verification of Large Language Model Generated Code Clones

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Verification Status: 100% Passed](https://img.shields.io/badge/Audit-1296%2F1296%20Cohorts%20Verified-brightgreen.svg)]()
[![Artifact DOI](https://img.shields.io/badge/Zenodo%20Raw%20Data-10.5281%2Fzenodo.14968618-blue.svg)](https://doi.org/10.5281/zenodo.14968618)

Official research replication repository prepared for:
> **"Reference-Guided Test Subsampling for Behavioral Verification of Large Language Model Generated Code Clones"**  
> *Target Journal: The Journal of Systems and Software (JSS), Elsevier.*  
> *Status: In Preparation for Submission (Pre-submission Research Artifact Package).*  
> *Official Public Repository: https://github.com/wintergithubdot/rtgs-data*

---

## 📌 Executive Overview

Evaluating the functional equivalence of Large Language Model (LLM) generated code typically requires executing extensive test suites (e.g., 256 inputs per benchmark problem). Across diverse prompts, temperatures, and model architectures, executing all candidate solutions against complete test suites introduces substantial computational overhead.

**Reference-Guided Test Subsampling (RGTS / M1)** is a candidate-blind test selection methodology that:
1. **Never accesses candidate code or executions** during selection (preventing data leakage and label contamination).
2. **Stratifies test inputs** into semantic output classes using the trusted reference implementation.
3. **Maximizes structural input diversity** within each semantic stratum using farthest-first traversal across 7 candidate-blind feature dimensions.

---

## 📊 Empirical Findings Summary ($N = 36$ Problems, $1,296$ Cohorts, $25,920$ Candidates)

All metrics below are computed with problem-level aggregation ($N = 36$ independent units of analysis) matching the manuscript tables and the audited Excel ledger (`results/03_Data_Ledger_Empiris_B3D_vs_M1.xlsx`):

### 1. Multi-Budget Verification Performance at Operational Budget $k = 32$ (87.5% Reduction)

| Evaluation Metric | Baseline B0 (Full Suite, $k=256$) | Uniform Random B1 ($k=32$) | Input-Stratified B2 ($k=32$) | Deterministic Input Diversity B3-D ($k=32$) | Proposed RGTS M1 ($k=32$) | Statistical Test vs M1 ($k=32$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test Suite Size ($k$)** | 256 | 32 | 32 | 32 | **32 (87.5% savings)** | — |
| **False Equivalence Rate (FER)** | Baseline (0.0%) | 1.93% $\pm$ 5.48% | 2.47% $\pm$ 9.78% | 1.05% $\pm$ 3.58% | **0.63% $\pm$ 3.06%** | $p_{\text{1s}} = 0.0108, p_{\text{budget}} = 0.0434$ (vs B1) |
| **Relative FER Reduction vs B1** | — | Baseline | -28.0% | +45.6% | **+67.4% error reduction** | $r_{\text{rb}} = +0.6340, W/T/L = 14/19/3$ |
| **Decision Agreement (DAR)** | 100.0% (Silver) | 97.45% $\pm$ 4.52% | 96.88% $\pm$ 4.71% | 99.13% $\pm$ 3.24% | **99.16% $\pm$ 3.38%** | $98.91\%$ on valid executing candidates |
| **Clustering Agreement (ARI)** | 1.0000 (Gold) | 0.8842 $\pm$ 0.1412 | 0.8915 $\pm$ 0.1380 | 0.9871 $\pm$ 0.0277 | **0.9880 $\pm$ 0.0294** | 0.9880 $\pm$ 0.0745 across 1,296 cohorts |
| **Active Strata Coverage** | 36 / 36 (100%) | Starvation Risk | Starvation Risk | 31 / 36 (86.1%) | **36 / 36 (100.0%)** | 100% representation guaranteed |
| **Runtime Acceleration** | $1.00\times$ | $5.52\times$ | $5.31\times$ | $5.50\times$ | **$5.52\times$ speedup** | Measured in test evaluation & clustering procedure |

*Statistical Note on B3-D vs M1*: As transparently reported in Section 5.2 and Table 3 of the manuscript, B3-D and M1 achieve identical FER on 33 of 36 problems ($W/T/L = 2/33/1$, dynamic exact permutation test $p_{\text{exact}} = 2/8 = 0.2500$, not statistically significant). RGTS eliminates localized strata starvation on 5 boundary problems (e.g., reducing FER from 11.07% to 0.00% on `min_pos_sum_subarr`).

### 2. Independent Clustering Implementation Equivalence ($N = 1,296$ Cohorts)

To guarantee that our representative clique clustering matches Eagal et al.'s baseline algorithm:
- **Exact Partition Matches under Fixed Denominator $|T_k|$ with Missing-Key Safeguard**: **1,292 / 1,296 cohorts (99.69%)** (the 4 partial-output cohorts evaluate with identical ARI = 1.000000).
- **Exact Partition Matches under Common Keys (`common_keys`)**: **1,296 / 1,296 cohorts (100.00%)**.
- **Maximum ARI Difference between Implementations**: **0.000000** across all 1,296 cohorts.

---

## 🚀 Quickstart (3-Minute Setup & Fixture Verification)

### 1. Installation
```bash
git clone https://github.com/wintergithubdot/rtgs-data.git
cd rtgs-data
pip install -r requirements.txt
```

### 2. Run Automated Unit & Integration Tests
```bash
pytest tests/ -v
# or:
python -m unittest discover tests
```
All 11 unit & integration tests (feature extraction, budget adherence, candidate-blindness, clique consistency, and pipeline integration) pass in under 2 seconds.

### 3. Verify Clustering Equivalence on Benchmark Fixtures
```bash
python scripts/verify_clustering_equivalence.py
```
This executes:
- **Level 1**: Synthetic transitive chain separation test.
- **Level 2**: Live clustering equivalence on 5 real benchmark cohorts from `fixtures/real_benchmark_cohorts_fixture.json`.
- **Level 3**: Dynamic census and cohort ledger recalculation across 25,920 candidates and 1,296 cohorts.

### 4. Reproduce Empirical Tables & Statistical Tests
```bash
python scripts/reproduce_results.py
```
This dynamically computes:
- **Table 1**: Candidate Execution Census & Sensitivity Analysis.
- **Table 2**: Paired Wilcoxon Signed-Rank Tests ($W/T/L$, $W^+$, $W^-$, $r_{\text{rb}}$, $p_{\text{1s}}$, $p_{\text{budget}}$, $p_{\text{global}}$) across 36 problems.
- **Table 3**: Matched-Budget Deterministic Head-to-Head Ablation (B3-D vs M1 at $k=32$, dynamic exact permutation $p_{\text{exact}} = 0.2500$).

---

## 🔬 Full Live Recomputation (All 1,296 Cohorts from Raw Data)

To independently extract every candidate execution trace and recompute clustering partitions from the raw benchmark data:

1. **Obtain raw benchmark dataset** from Eagal et al. (2025):
   - **Zenodo DOI**: [10.5281/zenodo.14968618](https://doi.org/10.5281/zenodo.14968618)
   - **File**: `data.zip` (available as a distinct standalone download on the Zenodo record; note that Eagal's baseline code is provided separately in `analyzing_the_dependability_of_large_language_models_for_code_clone_generation.zip`).
   - **MD5 Checksum of data.zip**: `d12c316e90d2de43501e0a57a74bcbed`
   - *Provenance Notice*: The benchmark dataset and baseline code are third-party artifacts created by Eagal et al. (2025). The RGTS repository contains our independent RGTS method implementation, verification scripts, replication fixtures, and empirical ledgers.

2. **Run live recomputation**:
```bash
python scripts/verify_clustering_equivalence.py --full-1296 --data-zip /path/to/data.zip
```
- **Runtime**: Approximately 1,114.7 seconds (18 minutes 35 seconds) for full live clustering extraction and evaluation across all 1,296 cohorts (total end-to-end script duration: 1,124.6 seconds / 18 minutes 44 seconds on an Intel Core i7 / Windows 64-bit benchmark machine).
- **Output**: Generates full cohort-by-cohort audit report matching `results/audit_clustering_1296_cohorts.csv` with 100.0% row-by-row consistency (exit code 0).
- **Fail-Fast Guard**: If `--full-1296` is called without a valid `data.zip`, the script terminates immediately with an informative message and exit code 1.

---

## 📁 Repository Structure

```
RGTS/
├── README.md               # Overview, quickstart, and reproduction guide
├── LICENSE                 # MIT Open Source License
├── CITATION.cff            # Formal citation metadata
├── requirements.txt        # Minimal Python dependencies (numpy, pandas, scikit-learn, scipy, pytest)
├── .gitignore              # Ignores large raw archives, caches, virtual environments
├── src/                    # Core library modules
│   ├── __init__.py         # Package initialization
│   ├── clustering.py       # Union-Find and representative-based clique clustering
│   ├── feature_encoder.py  # 7-dimensional input feature extraction & semantic strata
│   ├── rgts_selectors.py   # Selectors for Baselines (B0-B4) and Proposed RGTS (M1)
│   ├── pipeline.py         # High-level RGTSPipeline interface
│   ├── evaluator.py        # Behavioral equivalence & ARI agreement metrics
│   ├── rgts_io.py          # Zip archive data loader
│   └── eagal_baseline_get_clusters.py # Clean reference clustering baseline
├── tests/                  # Automated test suite
│   ├── __init__.py
│   ├── test_all.py         # Core unit tests (features, budgets, anti-leakage, clique)
│   ├── test_synthetic_equivalence.py # Level 1 transitive chain separation test
│   ├── test_feature_encoding.py      # Feature dimensions and strata classification
│   └── test_pipeline_integration.py  # End-to-end integration and key fidelity test
├── scripts/                # Verification and replication scripts
│   ├── verify_clustering_equivalence.py # Three-level verification suite & live recomputation
│   └── reproduce_results.py            # Reproduces Table 1, Table 2, and Wilcoxon tests
├── fixtures/               # Included benchmark subsets
│   └── real_benchmark_cohorts_fixture.json # 5 full/partial cohorts for instant testing
├── results/                # Complete empirical ledgers
│   ├── audit_census_25920_candidates.csv # Trace status of all 25,920 candidates
│   ├── audit_clustering_1296_cohorts.csv # Metrics for all 1,296 evaluated cohorts
│   ├── 03_Data_Ledger_Empiris_B3D_vs_M1.xlsx # 9-sheet empirical workbook
│   └── live_1296_cohorts_verification_log_NEW.txt # Live verification terminal log
└── docs/                   # Detailed technical documentation
    ├── experiment_protocol.md # 7 features, quotas, and farthest-first protocol
    └── reproducibility.md     # In-depth reproduction guide & verification tiers
```

---

## 📚 Citation

```bibtex
@article{triyono2026rgts,
  title     = {Reference-Guided Test Subsampling for Behavioral Verification of Large Language Model Generated Code Clones},
  author    = {Triyono, Andri and Sediyono, Eko and Hartomo, Kristoko Dwi},
  journal   = {The Journal of Systems and Software},
  publisher = {Elsevier},
  year      = {2026},
  note      = {In Preparation for Submission},
  url       = {https://github.com/wintergithubdot/rtgs-data}
}
```
