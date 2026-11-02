# RGTS Experimental Protocol

This document outlines the formal algorithmic specification and experimental setup for **Reference-Guided Test Subsampling (RGTS)** as submitted to the *Journal of Systems and Software* (Elsevier).

---

## 1. Candidate-Blind Architecture

In behavioral equivalence evaluation of LLM-generated code clones, candidate programs $P_1, P_2, \dots, P_M$ are untrusted and potentially faulty. A fundamental design constraint is that test selection must never depend on candidate execution outputs:
$$\text{Test Selection} \perp P_i \quad \forall i \in \{1, \dots, M\}$$

RGTS utilizes solely:
1. Candidate-blind structural input features ($\mathbf{x}_t \in \mathbb{R}^7$).
2. The trusted reference implementation behavior ($y^{\text{ref}}_t \in \mathcal{Y}$).

---

## 2. Seven Structural Input Features

For each test input $t$ in the benchmark pool ($N = 256$ inputs):
1. **Character Length** ($\log(1 + \text{length})$): Captures raw input payload size.
2. **Token Count** ($\log(1 + \text{tokens})$): Structural token count separated by whitespace, commas, and delimiters.
3. **Presence of Negative Numbers** ($\mathbb{I}[-\in t]$): Indicator for negative scalar values or sign boundaries.
4. **Presence of Zero** ($\mathbb{I}[0 \in t]$): Indicator for zero values, identity elements, or null transitions.
5. **Maximum Numeric Magnitude** ($\log(1 + \max |v|)$): Extreme numerical value stress testing.
6. **Structural Nesting Depth** ($\mathbb{I}[\text{nested}]$): Indicator for nested multi-dimensional arrays or collections.
7. **Character Entropy** ($H(t) = -\sum p_c \log_2 p_c$): Information density and distribution of characters.

Each feature dimension is Z-score standardized across the test suite pool:
$$z_{t, d} = \frac{x_{t, d} - \mu_d}{\sigma_d}$$
where degenerate features with $\sigma_d < 10^{-8}$ are safely clamped to $z_{t, d} = 0$.

---

## 3. Reference Output Stratification (Semantic Strata)

Reference outputs $y^{\text{ref}}_t$ are partitioned into 6 mutually exclusive semantic partitions:
- **Class 0**: Empty, null, false, 0, or trivial identity outputs.
- **Class 1**: Boolean True or scalar positive affirmation.
- **Class 2**: Single numeric scalars (integers, floats).
- **Class 3**: Short sequences / collections ($\le 5$ items).
- **Class 4**: Long sequences / structured nested outputs ($> 5$ items).
- **Class 5**: Complex string representations or unclassified outputs.

Each stratum $S_c$ receives a proportional test budget quota for target budget $k$ (e.g., $k = 32$):
$$q_c = \max\left(1, \left\lfloor \frac{k \cdot |S_c|}{N} + 0.5 \right\rfloor\right)$$
with proportional adjustment ensuring $\sum_c q_c = k$.

---

## 4. Farthest-First Input Traversal within Strata

Within each stratum $S_c$, inputs are selected iteratively using farthest-first traversal on normalized input feature vectors:
1. **Initial Seed**: Select the test input furthest from the stratum centroid to capture edge cases.
2. **Greedy Expansion**: Iteratively add test $t^* \in S_c \setminus \mathcal{T}_c$ maximizing minimum distance to current selections:
   $$t^* = \arg\max_{t \in S_c \setminus \mathcal{T}_c} \min_{u \in \mathcal{T}_c} \|\mathbf{z}_t - \mathbf{z}_u\|_2$$
3. Repeat until quota $q_c$ is satisfied.

This ensures maximal coverage of behavioral partitions while capturing input domain diversity without leaking candidate information.

