# 1. Core Results

**CR1 | Single Backbone Baselines**

* What: Train ViT-B/16, Swin-T, ConvNeXt-T on each dataset (standard aug).
* Metrics: AUROC/AUPRC, F1, ECE, Params, FLOPs.
* Threshold: Reproduce literature/official baselines within ±1%; run 5 seeds and report mean±std.

**CR2 | Equivariant Adapter PREA (Plug-and-play)**

* What: Add PREA to ViT-B/16 (inserted before/after patch embed and before MHA), two settings:
    * Frozen backbone + train only PREA (low-rank / lightweight).
    * Full-model finetune (baseline).
* Metrics: same as CR1 plus $\Delta_{\text{rot}}$ (rotation consistency).
* Threshold: On ISIC and CheXpert increase AUROC ≥ +1.0%, and reduce $\Delta_{\text{rot}}$ by ≥ 20%.

**CR3 | Multi-model Sources (Ensemble Material)**

* What: For each dataset train 3–5 finetuned models (different hospitals / different folds / different seeds).
* Metrics: Individual model AUROC and variance.
* Threshold: Create performance diversity (max - min ≥ 2% AUROC) to enable effective fusion.

**CR4 | OPM (Permutation + Orthogonal) Alignment + Weight Fusion (POPMI-Soup)**

* What: First perform re-basin permutation alignment, then orthogonal Procrustes alignment, then linear/weighted model soup → single weight.
* Comparisons:
    * Direct model soup (no alignment)
    * Git Re-Basin (permutation only, no orthogonal)
    * OT-fusion / FedMA style methods
    * Reference equivariant backbones (e.g., GE-ViT / E(2)-ViT)
* Metrics: AUROC/AUPRC, $\Delta_{\text{rot}}$, inference latency/throughput, Params.
* Threshold: On average across 3 datasets achieve +1.0% AUROC vs best baseline; inference cost ≈ single model (much lower than ensemble).

**CR5 | End-to-end PREA + OPM (POPMI full)**

* What: For CR3's multiple models train PREA on each, then perform OPM alignment and fusion.
* Threshold: Further reduce $\Delta_{\text{rot}}$ by ≥ 10% vs CR4, and small additional AUROC improvement (≥ +0.3%).

---

# 2. Equivariance & Robustness (Rotation & Robustness)

**RR1 | Controlled Rotation Sweep**

* What: For each test image rotate by θ ∈ {0,10,20,...,350°}, measure output differences and performance curves.
* Metrics: $\Delta_{\text{rot}}$ (logit/softmax difference), AUROC@θ, curve area.
* Threshold: POPMI's curve fluctuation ≤ 60% of baseline.

**RR2 | Random Angles & Composite Corruptions**

* What: Random angle + Gaussian blur / brightness-contrast / mild affine; observe robustness.
* Metrics: AUROC drop, ECE change.
* Threshold: POPMI's performance drop ≤ baseline's drop by ≥ 30%.

**RR3 | Data Augmentation Sensitivity**

* What: Compare three training configs: no rotation augmentation / moderate / strong rotation augmentation.
* Threshold: POPMI maintains low $\Delta_{\text{rot}}$ under "no/moderate" settings.

---

# 3. Fusion & Matching Analysis

**FM1 | Alignment Strategy Ablation**

* What:
    * Permutation only (P)
    * Rotation (orthogonal) only (R)
    * P+R (OPM)
    * OT (Sinkhorn) soft matching vs Hungarian hard matching
* Metrics: interpolation curve loss barrier, final AUROC, weight distance before/after alignment.
* Threshold: P+R yields lowest barrier and highest AUROC.

**FM2 | Layer & Granularity**

* What: Align only MLP projection layers, only attention projection / heads, or full-layer; head-level vs channel-level.
* Threshold: Identify the best cost-performance granularity (best performance with minimal compute).

**FM3 | Fusion Weighting Strategies**

* What: Uniform averaging, validation-weighted, greedy soup, Bayesian-type weights.
* Threshold: Greedy or validation weighting outperforms uniform by ≥ 0.3% AUROC.

**FM4 | Linear Path Connectivity & Barriers**

* What: Evaluate loss/acc along 20 points on W1 ↔ Ŵ2 linear path; compute barrier height.
* Threshold: After OPM the path is smoother (barrier reduced by ≥ 30%).

---

# 4. PREA Module Ablations (Adapter Design)

**PA1 | Insertion Positions**: only at patch embed, only before MHA, or both.
**PA2 | Rotation Group Discretization**: |Θ| = 4, 12, 24 (every 90°/30°/15°) — observe performance vs cost.
**PA3 | Positional Encoding**: absolute / relative / group-equivariant positional encodings comparison.
**PA4 | Training Scope**: frozen backbone vs unfreeze first K layers vs full unfreeze.
**Metrics/Threshold**: Minimize $\Delta_{\text{rot}}$ and maintain or improve AUROC without significantly increasing FLOPs (< +5%).

---

# 5. Cross-domain & OOD Generalization

**GD1 | Leave-one-hospital-out (CheXpert)**

* What: Simulate multi-hospital (or multi-era) splits, hold out one hospital/time for test; POPMI fuses models trained on the remaining hospitals.
* Threshold: ≥ +1.0% AUROC vs single-model average.

**GD2 | Dataset Transfer (ISIC 2018 → 2020)**

* What: Train on 2018, test on 2020; also perform reverse.
* Threshold: POPMI reduces transfer gap by ≥ 25%.

**GD3 | Low-shot Scenarios (MedMNIST v2)**

* What: Train with 1%/5%/10% samples per task; multi-models come from different subsets and apply OPM-soup.
* Threshold: In 1–5% settings POPMI > baseline by ≥ +2% AUROC.

---

# 6. Efficiency & Resources

**EF1 | Inference Cost**: single model vs ensemble vs POPMI (single weight).

* Metrics: latency (ms/image), throughput (img/s), GPU/CPU measurements, memory footprint, Params, FLOPs.
* Threshold: POPMI ≈ single model; significantly better than ensemble (latency reduction ≥ 40%).

**EF2 | Training / Alignment Overhead**: record OPM alignment time and finetune cycles after alignment.

* Threshold: alignment time controllable within < 10–20% of a single finetune cycle.

---

# 7. Theoretical Checks

**TH1 | Empirical Equivariance Error Bounds**

* What: Estimate ε(θ) and compare to theoretical upper bounds (using Lipschitz estimates / residual terms), plot angle vs error-bound curves.
* Threshold: Trends agree; POPMI empirical errors lie within or near the bound.

**TH2 | Interpolation Loss Upper Bound After Alignment**

* What: Interpolate with α ∈ [0,1], verify L(Ŵ) is not above the endpoint maximum plus theoretical term; regress residuals.
* Threshold: R² ≥ 0.8 (explained variance of the bound).

---

# 8. Interpretability & Failure Analysis

**DG1 | Attention & Feature Alignment Visualizations**: attention-head similarity (CKA/CCA), t-SNE before/after alignment.
**DG2 | Grad-CAM / Attention Rollout**: check whether focus regions are consistent before/after rotation.
**DG3 | Error-Case Clusters**: cluster misclassified samples and inspect effects related to equivariance (e.g., directional lesions).

---

# 9. Statistical Tests & Reproducibility

**ST1 | Significance**: AUROC with DeLong / bootstrap 95% CI; apply Holm-Bonferroni correction across tasks.
**ST2 | Multiple seeds (≥5)**: fix all randomness sources; report mean±std.
**ST3 | Controlled settings**: fix input size, training epochs, optimizer (AdamW), lr/WD/augmentation to ensure comparability.

---

# 10. Datasets & Splits (Practical)

* MedMNIST v2: use official train/val/test; additionally build a rotated test set.
* CheXpert: official train/val; create hospital/year splits if metadata available; report multi-label macro/micro AUROC.
* ISIC 2018 / 2020: official splits; add cross-year transfer evaluation.
* Common preprocessing: resize to 224/256 and center-crop; for rotation experiments avoid random cropping to prevent border artifacts.

---

# 11. Figures for the Paper (you can generate these when you run experiments)

* Fig.1: POPMI method diagram (PREA + OPM).
* Fig.2: Rotation robustness curves (RR1/RR2).
* Fig.3: Main results table (CR1–CR5).
* Fig.4: Alignment / fusion ablations (FM1–FM3).
* Fig.5: Linear-path loss barrier (FM4).
* Fig.6: Efficiency & resources (EF1/EF2).
* Fig.7: Theoretical checks (TH1/TH2).
* Appendix: visualizations and failure analysis (DG1–DG3), extra low-shot / cross-domain details (GD1–GD3).

---

## Success / Fallback Submission Strategies (Execution)

* Success path: CR5 significantly beats GE-ViT and fusion baselines; RR1/2 curves are clean; EF1 shows single-model cost.
* Fallback path: If equivariance gains are limited, emphasize "OPM-soup's robustness advantages in low-shot / cross-domain settings" (GD3/GD1) and target JBHI / CMPB or MICCAI/ISBI workshops.
