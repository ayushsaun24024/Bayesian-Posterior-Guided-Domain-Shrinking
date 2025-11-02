# Bayesian Posterior‑Guided Domain Shrinking (BPG‑DS) vs REDS

Implementation and experimental evaluation of efficient Bayesian optimization with domain shrinking, comparing deterministic (REDS) and probabilistic (BPG‑DS) pruning strategies.

---

## Project Documentation

This repository contains the complete research documentation, implementation, and experimental results:

- **BML_AYUSH_SAUN_MT24024.pdf**: Original project proposal outlining the research motivation, novelty, and feasibility of the Bayesian posterior‑guided approach.
- **BML_Project_interim_report.pdf**: Research findings on existing methods (GP‑UCB, Thompson Sampling, REDS) and the identified gap motivating BPG‑DS.
- **main_paper.pdf**: REDS algorithm paper (Salgia et al., ICML 2024), the baseline method and theoretical foundation.
- **bml-project.ipynb**: Complete modular notebook implementation with all methods, utilities, benchmarks, and experiments.

---

## 1) Problem Statement

Expensive black‑box objectives require sample‑efficient optimization. A Gaussian Process (GP) surrogate models the objective; the search domain is iteratively reduced to focus evaluations on promising regions.

- Baseline (REDS): random exploration with deterministic domain shrinking (UCB ≥ l*).
- Proposed (BPG‑DS): random exploration with probabilistic domain shrinking using GP posterior samples.

Goal: evaluate whether probabilistic pruning improves optimization quality while retaining practical efficiency.

---

## 2) Repository Structure

The notebook enforces a fixed cell layout:
1. Package installation
2. Imports
3. CONFIG (all hyperparameters)
4. Utilities
5. Mathematical functions
6. GP model
7. REDS algorithm
8. Benchmark functions
9. Visualization
10. REDS run example
11. BPG‑DS algorithm
12. BPG‑DS comparison runner
13. Benchmark experiment runner
14. Benchmark suite execution

All parameters controlled via CONFIG. No hidden constants.

---

## 3) Methods Implemented

### 3.1 REDS (Baseline)
- Epoch‑based random exploration with doubling schedule.
- Information‑theoretic β_n (incorporates information gain γ_n, RKHS bound B, confidence δ).
- Deterministic pruning: keep x if UCB(x) ≥ max(LCB) over the active domain.

### 3.2 BPG‑DS (Proposed)
- GP posterior sampling via Cholesky decomposition.
- Posterior probability p(x) = fraction of samples where x is within ε of each sample's maximum.
- Probabilistic pruning: keep x if p(x) ≥ threshold.

Only the pruning rule differs; all other components remain identical to the baseline.

---

## 4) Test Functions and Domains

Implemented benchmark objectives with configurable bounds and grid sizes in CONFIG:
- Branin (2D)
- Hartmann‑3D (3D)
- Hartmann‑6D (6D; very large grid, see practical notes)
- Ackley‑2D (2D)
- Rosenbrock‑2D (2D)

No external datasets required.

---

## 5) Utilities and Preprocessing

- Domain discretization, random sampling, seeding.
- RBF kernel, UCB/LCB computation, information‑theoretic β_n.
- GP wrapper (fit, predict, update) with numerical stability.
- Regret metrics: simple and cumulative.
- Results aggregation across seeds.

---

## 6) Experiments Run

- 10 random seeds per function.
- Fixed budget with 6 epochs; identical initialization for fair comparison.
- Metrics: best value, simple regret, cumulative regret, runtime, domain size.
- Visualizations: convergence curves, regret bands (mean ± std), domain shrinking, summary bar charts, posterior probability heatmaps (2D).

---

## 7) Results Summary

**Branin (2D)**
- REDS: simple regret 0.1008 ± 0.0968
- BPG‑DS: simple regret 0.1712 ± 0.1547
- Outcome: baseline better on this function.

**Hartmann‑3D (3D)**
- REDS: simple regret 0.1628 ± 0.0730
- BPG‑DS: simple regret 0.0730 ± 0.0529
- Outcome: proposed method substantially better (+55.2%).

**Ackley‑2D (2D)**
- REDS: simple regret 4.4421 ± 1.1881
- BPG‑DS: simple regret 1.7758 ± 0.8994
- Outcome: proposed method substantially better (+60.0%).

**Rosenbrock‑2D (2D)**
- REDS: simple regret 0.0308 ± 0.0349
- BPG‑DS: simple regret 0.0947 ± 0.1331
- Outcome: baseline better on this function.

**Timing**
- On 2D grids, runtimes are comparable per seed.
- On large/high‑D grids, posterior sampling can be costly; avoid extremely large discretizations.

---

## 8) How to Reproduce

1. Open the main notebook (bml-project.ipynb) in Colab/Kaggle/local environment.
2. Run cells in order. The notebook structure enforces consistent ordering.
3. Adjust CONFIG:
   - Function list, seeds, epochs
   - Grid sizes and bounds
   - BPG‑DS: K_SAMPLES, P_THRESHOLD, EPSILON
4. Run:
   - Individual method examples
   - REDS vs BPG‑DS comparison
   - Full benchmark suite
5. Outputs:
   - Plots rendered in‑notebook
   - Printed summary tables for each benchmark

---

## 9) Practical Notes

- For higher dimensions, reduce grid size or subsample the domain before posterior sampling.
- Use the same seeds and budgets for fair method comparison.
- Interpret results per function type:
  - Multimodal/complex (e.g., Hartmann‑3D, Ackley‑2D): probabilistic pruning often helps.
  - Simple/narrow‑valley (e.g., Branin, Rosenbrock‑2D): deterministic pruning can be stronger.
- Hartmann‑6D with full grid (1M points) is computationally prohibitive for BPG‑DS; use coarser discretization or domain subsampling if necessary.

---

## 10) Changelog

- Week 1: Baseline REDS implemented and validated on Branin and Hartmann‑3D.
- Week 2: BPG‑DS implemented; posterior sampling and probabilistic pruning added.
- Week 3–4: Benchmark suite, aggregation, and visualizations completed for Branin, Hartmann‑3D, Ackley‑2D, Rosenbrock‑2D. Hartmann‑6D attempted; limited by domain size.

---

## 11) References

- REDS paper: S. Salgia, V. Vakili, Q. Zhao, "Random Exploration in Bayesian Optimization: Order‑Optimal Regret and Computational Efficiency," ICML 2024.
- Project proposal: BML_AYUSH_SAUN_MT24024.pdf (motivation, novelty, feasibility).
- Research findings: BML_Project_interim_report.pdf (literature review, gap identification).

---

## 12) License

This repository is released for academic and non‑commercial use. See LICENSE for details.
