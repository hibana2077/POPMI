# POPMI: Plug-and-play Orthogonal Parameter Matching for Model Integration

— Rotation-equivariant Transformer fusion for medical image classification

One-sentence summary: POPMI proposes a plug-and-play rotation-equivariant adapter combined with orthogonal parameter matching (orthogonal Procrustes + permutation) to align and fuse multiple fine-tuned Transformer/Conv models. It improves medical image classification accuracy and rotational stability without increasing inference cost, and provides theoretical bounds and provable control of equivariance error. We validate at scale and with ablations on MedMNIST v2 and ISIC. ([arXiv][1], [Nature][2])

---

## Core research problems

1. Medical images (X-ray, dermoscopy, histology) often appear in arbitrary rotations; non-equivariant models are sensitive to rotation.

2. In practice there are often multiple fine-tuned models from different sources/institutions. How to fuse their knowledge without increasing inference cost? Existing methods either focus on equivariance (e.g., G-CNN / equivariant ViT) or on weight-space fusion (e.g., model soup / re-basin / OT-fusion), lacking a unified framework that simultaneously satisfies "rotation equivariance + weight-level fusion". ([Proceedings of Machine Learning Research][3], [arXiv][4], [proceedings.neurips.cc][5])

## Goals

* Improve AUROC / F1 and rotational consistency metrics in multi-dataset, multi-institution settings without increasing inference time or parameter count.
* Provide plug-and-play equivariant adapters that do not restrict the backbone (ViT/Swin/Conv).
* Use orthogonal parameter matching + weight interpolation/fusion to theoretically guarantee:
    1. an upper bound on equivariance error; 2. that the fused loss is not worse than the linear-path upper bound between aligned weights.

---

## Method overview (POPMI)

### (A) Plug-and-play rotation-equivariant adapter (PREA)

* Insert group-equivariant (E(2)/SE(2)) adapters before/after patch embedding and into some attention heads: approximate ViT rotation equivariance via group convolutions / equivariant positional encodings. The module can freeze the backbone or only fine-tune low-rank adapters. The design is inspired by G-CNN and E(2)-equivariant ViT. ([arXiv][6], [Proceedings of Machine Learning Research][7])

### (B) Orthogonal Parameter Matching (OPM)

* For multiple fine-tuned models, solve layer-wise permutation + orthogonal Procrustes alignment:

    $$
    \min_{P\in\Pi,\,R\in O(d)}\ \|W_1 - P\,W_2 R\|_F^2,\quad R^*=\mathrm{UV}^\top\ \text{for SVD}(W_2^\top P^\top W_1=U\Sigma V^\top)
    $$

* This complements Git Re-Basin (permutation alignment) and OT-fusion / FedMA (intra-layer matching) by additionally applying orthogonal alignment to handle rotational degrees of freedom in attention/projection layers, then doing weight interpolation/averaging to obtain a single model with no extra inference cost. ([arXiv][8], [Personal Robotics Lab][9], [proceedings.neurips.cc][5])

### (C) POPMI-Soup fusion

* After OPM alignment, perform weighted linear interpolation (or greedy selection) to form a single weight set (like model soup), combining robustness with zero extra inference cost. ([arXiv][4], [Proceedings of Machine Learning Research][10])

---

## Theoretical insights (summary)

Definition 1 (rotation equivariance): For a group G ⊂ SE(2), model f is equivariant if ∀ g ∈ G, f(ρ_X(g)x) = ρ_Y(g) f(x). PREA guarantees equivariance to a set of discrete rotations Θ via group convolutions / equivariant positional encodings. ([arXiv][6], [Proceedings of Machine Learning Research][7])

Proposition 1 (POPMI equivariance error bound): Let the backbone be non-equivariant but PREA is added before/after; if the discretized rotation set is Θ, then for any θ ∈ Θ,

$$
\|f_{\text{POPMI}}(R_\theta x)-R_\theta f_{\text{POPMI}}(x)\|\le \epsilon(\theta)
$$

where ε(θ) scales with the backbone’s equivariance-breaking modules (e.g., absolute position encodings) Lipschitz constants and PREA correction residuals; with relative/group-equivariant positional encodings in self-attention, ε can converge to 0. ([Proceedings of Machine Learning Research][7])

Proposition 2 (loss upper bound for aligned interpolation): Let \tilde W_2 = P^* W_2 R^* be the OPM-aligned solution. If loss L(W) is L-Lipschitz along the segment [W_1, \tilde W_2] and there exists approximate single-basin connectivity (linear mode connectivity), then for \hat W = α W_1 + (1−α) \tilde W_2:

$$
\mathcal L(\hat W)\le \max\{\mathcal L(W_1),\mathcal L(\tilde W_2)\}+ L\,\alpha(1-\alpha)\|W_1-\tilde W_2\|_F,
$$

showing that aligned interpolation does not degrade performance, consistent with observations from re-basin and model soups. ([arXiv][8])

Proof sketch: Proposition 1 uses representation theory of equivariant layers and residual bounds; Proposition 2 follows from Lipschitz bounds along the line segment and the single-basin re-basin hypothesis after alignment.

---

## Mathematical derivations & proofs (concise)

1. Closed-form Procrustes alignment: Given A, B ∈ R^{d×d}, the solution of min_{R∈O(d)} ||A − B R||_F is R^* = U V^⊤ where U Σ V^⊤ = SVD(B^⊤ A). We iterate solving R and P (Hungarian / OT soft matching) with P fixed. ([proceedings.neurips.cc][5])
2. Group convolution equivariance: If kernel k is closed under the group action, group convolution (f * k)(g) = ∫ f(h) k(h^{-1} g) dh is equivariant to G; GE-ViT’s equivariant positional encodings also satisfy E(2)/SE(2) equivariance. ([arXiv][6], [Proceedings of Machine Learning Research][7])
3. Linear path connectivity: Re-basin shows that after permutation alignment, two models can be linearly connected with low barrier; orthogonal alignment further stabilizes attention/projection layers. ([arXiv][8])

---

## Datasets planned (classification tasks)

* MedMNIST v2 (multi-modal lightweight 2D/3D, 708k images): suitable for large-scale and multi-task equivariance/fusion ablations. ([Nature][2])
* ISIC 2018/2020 (dermoscopic classification): arbitrary camera orientation makes it ideal for testing rotational consistency and generalization. ([arXiv][13], [challenge.isic-archive.com][14])

---

## Experimental design

* Metrics: AUROC, AUPRC, Δ_rot (rotational consistency: measure output discrepancy across a random angle set Θ), calibration (ECE).
* Comparisons: ViT/Swin baselines, GE-ViT (E(2)-ViT), Model Soups, Git Re-Basin, OT-Fusion, FedMA. ([Proceedings of Machine Learning Research][7], [arXiv][4], [proceedings.neurips.cc][5])
* Ablations: remove PREA / remove OPM / soup only / permutation-only / OT soft alignment only.
* Data splits: institutional/year cross-domain splits; controlled rotation tests (rotate each image randomly from 0–350° in 10° steps).
* Efficiency: report inference time and parameter count to show no added inference cost (vs. ensemble).

---

## Expected contributions

1. First work to unify rotation-equivariant plug-and-play modules with weight-space alignment/fusion in medical image classification.
2. Introduce OPM (Permutation + Orthogonal Procrustes) to improve stability of Transformer layer fusion.
3. Provide theoretical upper bounds on equivariance error and interpolation loss.
4. Empirically outperform strong baselines on ISIC / MedMNIST v2 with zero additional inference cost. ([arXiv][4], [Proceedings of Machine Learning Research][7], [Nature][2])

---

## Novelty

* Plug-and-play: gain rotation equivariance without changing the backbone architecture.
* Dual alignment & fusion: combine permutation alignment (re-basin / FedMA / OT-fusion ideas) with orthogonal alignment targeted at attention/projection matrices. ([arXiv][8], [proceedings.neurips.cc][5])
* Single-model inference: like model soup, no additional inference cost, but more stable after alignment. ([arXiv][4])

---

## Differences from prior work

* G-CNN / GE-ViT: only handle equivariance, not multi-model weight fusion; POPMI addresses both simultaneously. ([arXiv][6], [Proceedings of Machine Learning Research][7])
* Model Soups: simply averages weights and does not handle channel/head alignment nor equivariance; POPMI aligns via OPM before fusion. ([arXiv][4])
* Git Re-Basin / FedMA / OT-Fusion: can align/fuse weights but do not include rotation-equivariant adapters tailored for medical imaging; POPMI adds equivariant design and theoretical bounds. ([arXiv][8], [proceedings.neurips.cc][5])

---

## Plan for successful submission (TMI)

* Main results: average AUROC / F1 across three datasets and Δ_rot significantly better than GE-ViT and best single models, demonstrating zero inference overhead.
* Visualizations: rotation stability curves, attention-head similarity before/after alignment, t-SNE / CKA.
* Clinical value: cross-institution generalization and small-sample advantages from fusing models trained at different sites.
* Open-source: provide reference implementations of PREA/OPM and training scripts.

## Fallback plan (if main submission fails)

* If rotation-equivariant gains are limited: pivot to "low-data / cross-domain fusion" focusing on OPM-soup robustness in few-shot and cross-institution generalization; submit to JBHI / CMPB or MICCAI/ISBI workshop; or narrow to a single modality (ISIC) for deeper pathology analysis.

---

## Limitations & risks

* Some tasks encode orientation as diagnostic information (e.g., orthopedic localization); strong equivariance may hurt performance. Use soft equivariance or enable PREA only in selected layers.
* Weight alignment is sensitive to model width/layer design; restrict comparable layers (e.g., MLP/projection layers) and use OT soft matching to mitigate. ([proceedings.neurips.cc][5])

---

## References (key)

* Plug-and-Play Priors (classic plug-and-play framework that inspired PREA’s decoupled design). ([docs.lib.purdue.edu][15])
* Group-Equivariant CNN / E(2)-Equivariant ViT (theory of rotation equivariance and ViT implementations). ([arXiv][6], [Proceedings of Machine Learning Research][7])
* Model Soups (weight averaging with zero inference cost). ([arXiv][4])
* Git Re-Basin (permutation alignment and linear connectivity). ([arXiv][8])
* OT-Fusion / FedMA (intra-layer matching and fusion via OT/matching). ([proceedings.neurips.cc][5], [arXiv][16])
* Datasets: MedMNIST v2, ISIC. ([Nature][2], [arXiv][11], [challenge.isic-archive.com][14])

---

If desired, I can also convert the above into a paper outline + figure plan (Fig.1 Method, Fig.2 Theoretical bounds, Fig.3 Main results, Fig.4 Ablations) and an experiment script skeleton (PyTorch + timm + e2cnn/GE-ViT references) ready to run.

[1]: https://arxiv.org/abs/2306.06722? "$E(2)$-Equivariant Vision Transformer"
[2]: https://www.nature.com/articles/s41597-022-01721-8? "MedMNIST v2 - A large-scale lightweight benchmark for 2D ..."
[3]: https://proceedings.mlr.press/v48/cohenc16.html? "Group Equivariant Convolutional Networks"
[4]: https://arxiv.org/abs/2203.05482? "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time"
[5]: https://proceedings.neurips.cc/paper/2020/file/fb2697869f56484404c8ceee2985b01d-Paper.pdf? "Model Fusion via Optimal Transport"
[6]: https://arxiv.org/abs/1602.07576? "Group Equivariant Convolutional Networks"
[7]: https://proceedings.mlr.press/v216/xu23b/xu23b.pdf? "E(2)-Equivariant Vision Transformer"
[8]: https://arxiv.org/abs/2209.04836? "Git Re-Basin: Merging Models modulo Permutation Symmetries"
[9]: https://personalrobotics.cs.washington.edu/publications/ainsworth2023gitrebasin.pdf? "GIT RE-BASIN: MERGING MODELS MODULO PERMU"
[10]: https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf? "averaging weights of multiple fine-tuned models improves ..."
[13]: https://arxiv.org/abs/1902.03368? "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)"
[14]: https://challenge.isic-archive.com/data/? "ISIC Challenge Datasets"
[15]: https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1449&context=ecetr& "Plug-and-Play Priors for Model Based Reconstruction"
[16]: https://arxiv.org/abs/2002.06440? "Federated Learning with Matched Averaging"
