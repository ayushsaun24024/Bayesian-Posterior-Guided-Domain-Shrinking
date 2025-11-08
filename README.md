# Bayesian Posterior‑Guided Domain Shrinking (BPG‑DS) vs REDS

Implementation and experimental evaluation of efficient Bayesian optimization with domain shrinking, comparing deterministic (REDS) and probabilistic (BPG‑DS) pruning strategies.

---

## Project Documentation

This repository contains the complete research documentation, implementation, and experimental results:

- **BML_AYUSH_SAUN_MT24024.pdf**: Original project proposal outlining the research motivation, novelty, and feasibility of the Bayesian posterior‑guided approach.
- **BML_Project_interim_report.pdf**: Research findings on existing methods (GP‑UCB, Thompson Sampling, REDS) and the identified gap motivating BPG‑DS.
- **main_paper.pdf**: REDS algorithm paper (Salgia et al., ICML 2024), the baseline method and theoretical foundation.
- **bml-project.ipynb**: Complete modular notebook implementation with all methods, utilities, benchmarks, and experiments.
- **ablation_results_final.csv**: Comprehensive ablation study results (15 seeds × 9 configs × 4 functions = 540 experiments).

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
2. Imports (including `pandas`, `scipy.stats`, `csv`, `gc` for ablation study)
3. CONFIG (all hyperparameters + ablation parameters)
4. Utilities (regret metrics + **new**: CSV storage, statistical tests, confidence intervals)
5. Mathematical functions
6. GP model
7. REDS algorithm
8. Benchmark functions
9. Visualization (including **new**: ablation heatmaps, confidence interval plots, statistical comparison charts)
10. REDS run example
11. BPG‑DS algorithm
12. BPG‑DS comparison runner
13. **NEW: Memory‑efficient ablation study runner**
14. **NEW: Statistical analysis with significance testing**
15. Benchmark experiment runner
16. Benchmark suite execution

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
- Branin (2D) - grid_size: 50
- Hartmann‑3D (3D) - **grid_size reduced to 20** (Nov 8 update: memory optimization)
- Hartmann‑6D (6D; very large grid, see practical notes)
- Ackley‑2D (2D) - grid_size: 50
- Rosenbrock‑2D (2D) - grid_size: 50

No external datasets required.

---

## 5) Utilities and Preprocessing

- Domain discretization, random sampling, seeding.
- RBF kernel, UCB/LCB computation, information‑theoretic β_n.
- GP wrapper (fit, predict, update) with numerical stability.
- Regret metrics: simple and cumulative.
- Results aggregation across seeds.
- **NEW (Nov 8)**: CSV‑based storage functions, statistical significance tests (t-test, Wilcoxon, Cohen's d), confidence intervals (parametric + bootstrap), memory management (gc.collect).

---

## 6) Experiments Run

### 6.1 Initial Benchmark Suite (Nov 2)
- 10 random seeds per function.
- Fixed budget with 6 epochs; identical initialization for fair comparison.
- Metrics: best value, simple regret, cumulative regret, runtime, domain size.
- Visualizations: convergence curves, regret bands (mean ± std), domain shrinking, summary bar charts, posterior probability heatmaps (2D).

### 6.2 Enhanced Ablation Study (Nov 8) **NEW**
- **15 seeds per configuration** (increased from 10 for statistical robustness).
- **Memory‑efficient implementation**: incremental CSV writes, minimal RAM usage, garbage collection after each experiment.
- **Grid tested**: K ∈ {20, 35, 50}, P ∈ {0.2, 0.3, 0.4}, ε = 0.1.
- **Total experiments**: 4 functions × 9 configs × 15 seeds = **540 experiments**.
- **Statistical rigor**: t-tests, Wilcoxon tests, Cohen's d effect sizes, 95% confidence intervals.
- **Voting system**: Democratic selection of optimal hyperparameters across all functions.

---

## 7) Results Summary

### 7.1 Initial Results (Nov 2 - 10 Seeds)

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

### 7.2 **Ablation Study Results (Nov 8 - 15 Seeds)** **NEW**

**Recommended Global Configuration (Voting)**:
- K_SAMPLES: 20
- P_THRESHOLD: 0.2
- EPSILON: 0.1

**Per‑Function Optimal Configurations**:
- **Branin**: K=20, P=0.2 → regret 0.052 ± 0.073
- **Ackley2D**: K=50, P=0.2 → regret 1.186 ± 0.584
- **Rosenbrock2D**: K=20, P=0.2 → regret 0.068 ± 0.090
- **Hartmann3D**: K=35, P=0.4 → regret 0.049 ± 0.019

**Statistical Significance**:
- **Hartmann3D**: 44.0% improvement over REDS (p < 0.001, highly significant)
- **Rosenbrock2D**: 78.1% improvement over REDS (p < 0.001, highly significant)
- **Ackley2D**: -406% (catastrophic failure, investigation ongoing)
- **Branin**: -0.7% (no significant difference)

**Key Finding**: BPG‑DS demonstrates **function‑specific performance** - excels on Hartmann3D and Rosenbrock2D but underperforms on Ackley2D.

**Timing**
- On 2D grids, runtimes are comparable per seed.
- On large/high‑D grids, posterior sampling can be costly; avoid extremely large discretizations.

---

## 8) How to Reproduce

### 8.1 Basic Benchmarks (Nov 2 Work)
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

### 8.2 **Ablation Study (Nov 8 Work)** **NEW**
1. Ensure CONFIG contains ablation parameters:
   ```
   ABLATION_K_SAMPLES = 
   ABLATION_P_THRESHOLD = [0.2, 0.3, 0.4]
   ABLATION_EPSILON = [0.1]
   ABLATION_N_SEEDS = 15
   ABLATION_FUNCTIONS = ['branin', 'ackley2d', 'rosenbrock2d', 'hartmann3d']
   ABLATION_RESULTS_FILE = 'ablation_results_final.csv'
   ```

2. Run ablation study cell (Cell 13):
   ```
   results_file = run_memory_efficient_ablation_study(
       function_names=CONFIG.ABLATION_FUNCTIONS,
       n_seeds=CONFIG.ABLATION_N_SEEDS,
       results_file=CONFIG.ABLATION_RESULTS_FILE
   )
   ```

3. Analysis automatically runs (Cell 14):
   - Loads CSV results
   - Computes statistics (mean, std, confidence intervals)
   - Performs significance tests (t-test, Wilcoxon, Cohen's d)
   - Generates visualizations (heatmaps, sensitivity plots)
   - Voting system selects optimal config

4. Outputs:
   - `ablation_results_final.csv`: All 540 experimental results
   - Statistical summary tables with p-values
   - Heatmaps showing K vs P sensitivity per function
   - Parameter sensitivity curves with error bars
   - Voting‑based recommendation

**Expected Runtime**: 4-6 hours for full ablation (540 experiments).

**Memory Usage**: <500MB RAM (CSV storage prevents OOM).

---

## 9) Practical Notes

- For higher dimensions, reduce grid size or subsample the domain before posterior sampling.
- Use the same seeds and budgets for fair method comparison.
- Interpret results per function type:
  - Multimodal/complex (e.g., Hartmann‑3D, Rosenbrock‑2D): probabilistic pruning often helps.
  - Simple/narrow‑valley (e.g., Branin): deterministic pruning can be stronger.
  - Highly multimodal (e.g., Ackley‑2D): requires investigation (ongoing work).
- Hartmann‑6D with full grid (1M points) is computationally prohibitive for BPG‑DS; use coarser discretization or domain subsampling if necessary.
- **NEW**: Ablation results incrementally saved to CSV - safe to interrupt and resume.

---

## 10) Changelog

- **Week 1**: Baseline REDS implemented and validated on Branin and Hartmann‑3D.
- **Week 2**: BPG‑DS implemented; posterior sampling and probabilistic pruning added.
- **Week 3–4**: Benchmark suite, aggregation, and visualizations completed for Branin, Hartmann‑3D, Ackley‑2D, Rosenbrock‑2D. Hartmann‑6D attempted; limited by domain size.
- **Nov 2 (Week 5)**: Full benchmark comparison pushed with 10 seeds per function.
- **Nov 8 (Week 6)** **NEW**:
  - Implemented memory‑efficient ablation study framework
  - Extended seeds from 10 to 15 for statistical robustness
  - Added statistical significance testing (t-test, Wilcoxon, Cohen's d)
  - Implemented confidence intervals (parametric + bootstrap)
  - Created voting system for hyperparameter recommendation
  - Generated publication‑quality visualizations (heatmaps, error bars)
  - Reduced Hartmann-3D grid_size from 30 to 20 for memory efficiency
  - Completed 540 experiments across 4 functions
  - Discovered function‑specific performance patterns (BPG‑DS excels on Hartmann3D/Rosenbrock2D, struggles on Ackley2D)

---

## 11) References

- REDS paper: S. Salgia, V. Vakili, Q. Zhao, "Random Exploration in Bayesian Optimization: Order‑Optimal Regret and Computational Efficiency," ICML 2024.
- Project proposal: BML_AYUSH_SAUN_MT24024.pdf (motivation, novelty, feasibility).
- Research findings: BML_Project_interim_report.pdf (literature review, gap identification).

---

## 12) Known Issues and Future Work

**Current Limitations**:
- Ackley2D catastrophic failure (-406% vs REDS) - investigating probabilistic threshold calibration mismatch.
- Domain reset issue detected: posterior probabilities (0.0-0.1 range) don't match absolute threshold (0.3).
- Computational overhead: BPG‑DS 50x slower than REDS due to posterior sampling (K=50).

**Future Directions**:
1. **Threshold Calibration**: Implement information‑theoretic threshold based on mean + k×std instead of absolute value.
2. **Theoretical Analysis**: Derive regret bounds for BPG‑DS to justify threshold selection.
3. **Real‑World Application**: Test on ML hyperparameter tuning (Week 7 planned).
4. **Parallelization**: Implement multi‑core posterior sampling for Hartmann‑6D.
5. **Workshop Submission**: Target NeurIPS 2026 BayesOpt Workshop with completed ablation results.

---

## 13) License

This repository is released for academic and non‑commercial use. See LICENSE for details.
```

This complete README is now properly formatted in a single code block with all the November 8 updates included, properly escaped quotes, and all content preserved.
