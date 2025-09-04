# POPMI (Plug-and-play Orthogonal Parameter Matching & Integration)

Scaffold implementation for the POPMI research plan:

* PREA: plug-and-play rotation-equivariant adapters for generic vision backbones.
* OPM: permutation + orthogonal (Procrustes) parameter matching for model fusion.
* POPMI-Soup: aligned weight interpolation producing a single inference-time model.

## Repository Layout

```
src/
	dataset/isic2018.py        # ISIC 2018 parquet dataset loader
	models/prea.py             # PREA adapter modules
	alignment/opm.py           # Orthogonal parameter matching + fusion helpers
	metrics/rotation.py        # Rotation consistency (Delta_rot) metric
	training/train_classifier.py  # Baseline training script (ISIC example)
	training/fuse.py           # Alignment + fusion CLI (POPMI-Soup)
	training/eval_rotation.py  # Evaluate rotation consistency
```

## Install

```
pip install -r requirements.txt
```

## Train Baseline (ISIC 2018 example)

```
python -m src.training.train_classifier --data-root /path/to/data --epochs 10 \
	--model-name vit_base_patch16_224 --out-dir outputs/vit_b
```

## (Optional) Enable PREA Adapters

Edit `train_classifier.py` to wrap model after creation:

```python
from models.prea import attach_prea_to_vit
model = build_model(cfg)
model = attach_prea_to_vit(model, group_size=12, hidden_ch=32)
```

## Train Multiple Seeds / Variants

Repeat training with different seeds / data splits to obtain multiple `best.pt` checkpoints.

## Alignment + Fusion (POPMI-Soup)

```
python -m src.training.fuse --checkpoints outputs/vit_b_seed*/best.pt \
	--out fused/fused.pt --model-name vit_base_patch16_224
```

## Evaluate Rotation Consistency

```
python -m src.training.eval_rotation --ckpt fused/fused.pt --data-root /path/to/data \
	--angles 0 30 60 90 120 150 180
```

## Next Steps (Per Experiment Plan)
* Add MedMNIST & CheXpert loaders.
* Implement calibration (ECE), AUROC micro/macro, and statistical tests.
* Extend OPM to attention head granularity & soft (OT) matching.
* Add scripts for ablations (P only / R only / P+R).
* Implement theoretical bound estimation (empirical epsilon curves).

## License
TBD
