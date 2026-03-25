# Re-IQA + ARNIQA Multi-Distortion Encoder

## What this is

This is a modification of the official Re-IQA codebase (CVPR 2023) that
replaces the single noise-aware encoder with a **Multi-Distortion Encoder (MDE)**
— a system of four specialist encoders, each trained on a specific distortion
type, fused by a Mixture-of-Experts (MoE) gating network.

The specialist heads each learn an **ARNIQA-style distortion manifold** — a
geometrically organised embedding space where position corresponds to distortion
severity for their specific type.

---

## What files you are adding / changing

```
ReIQA/                                  ← original repo root
│
├── networks/
│   └── multi_distortion_encoder.py     ← NEW — the core module
│
├── moco/
│   ├── distortion_augmentations.py     ← NEW — per-distortion augmentations
│   ├── losses.py                       ← NEW — combined training loss
│   └── builder_mde.py                  ← NEW — MoCo wrapper for MDE
│
├── main_contrast_mde.py                ← NEW — training entry point
├── demo_quality_aware_feats_mde.py     ← NEW — feature extraction
│
│   (everything below is UNTOUCHED from original Re-IQA)
├── main_contrast.py                    ← original, untouched
├── demo_quality_aware_feats.py         ← original, untouched
├── demo_content_aware_feats.py         ← original, untouched
├── moco/builder.py                     ← original, untouched
├── moco/loader.py                      ← original, untouched
├── learning/                           ← original, untouched
└── networks/                           ← original, untouched
```

You are **adding 5 new files** and touching **nothing** in the original codebase.
The content encoder, regressor, and evaluation code are completely unchanged.

---

## Architecture overview

### Original Re-IQA
```
image
  ├──► Noise-aware encoder (1 x ResNet-50) ──► 128-dim quality embed ──┐
  └──► Content encoder    (1 x ResNet-50) ──► 2048-dim content embed ──┤
                                                                        ▼
                                                              concat → regressor → score
```

### Modified Re-IQA (this repo)
```
image
  ├──► Shared backbone (ResNet-50 layers 1-4) ──► 2048-dim feat
  │         ├──► Gaussian head ──► ARNIQA manifold G ──► f1 (128-dim)
  │         ├──► Blur head     ──► ARNIQA manifold B ──► f2 (128-dim)
  │         ├──► JPEG head     ──► ARNIQA manifold J ──► f3 (128-dim)
  │         └──► Weather head  ──► ARNIQA manifold W ──► f4 (128-dim)
  │
  ├──► Gating network (ResNet-18) ──► [w1, w2, w3, w4]  (sum to 1)
  │
  │    fused = w1*f1 + w2*f2 + w3*f3 + w4*f4  ──► 128-dim quality embed ──┐
  │                                                                          │
  └──► Content encoder (unchanged) ──────────────────► 2048-dim embed ──────┤
                                                                             ▼
                                                                   concat → regressor → score
```

---

## How the training works

### Phase 1 — Unsupervised pre-training (multi-distortion contrastive learning)

Run `main_contrast_mde.py`. Three losses train simultaneously:

| Loss | What it teaches | Weight |
|------|----------------|--------|
| InfoNCE | Distorted views of same image = similar embedding | 1.0 (fixed) |
| Manifold triplet | Mild distortion closer to clean than heavy distortion | `--lambda_triplet` (default 0.5) |
| Gating entropy | All 4 heads should be used, not just one | `--lambda_gate` (default 0.1) |

The content encoder is trained **separately** using original Re-IQA's
`main_contrast.py` — this is unchanged.

### Phase 2 — Supervised regressor training

Same as original Re-IQA:
1. Extract MDE quality features: `demo_quality_aware_feats_mde.py`
2. Extract content features: original `demo_content_aware_feats.py` (unchanged)
3. Concatenate: `combined = np.concatenate([quality_128, content_2048], axis=1)`
4. Train Ridge/ElasticNet regressor on labelled IQA datasets

---

## Installation

No new dependencies. Uses the same requirements as original Re-IQA:
```bash
pip install -r requirements.txt
```
The only requirement is `torchvision >= 0.14` (already in Re-IQA's requirements).

---

## Training commands

### Phase 1: Train multi-distortion encoder
```bash
python main_contrast_mde.py \
    --csv_path ./csv_files/moco_train.csv \
    --model_path ./expt_mde \
    --tb_path ./expt_mde \
    --batch_size 256 \
    --learning_rate 6.0 \
    --epochs 40 \
    --lambda_triplet 0.5 \
    --lambda_gate 0.1 \
    --cosine \
    --optimizer LARS
```

### Phase 1: Train content encoder (unchanged from original Re-IQA)
```bash
python main_contrast.py \
    --method MoCov2 --cosine --head mlp \
    --csv_path ./csv_files/moco_train.csv \
    --model_path ./expt_content \
    --optimizer LARS \
    --batch_size 630 --learning_rate 12 --epochs 40
```

### Phase 2: Extract features
```bash
# Quality features (new)
python demo_quality_aware_feats_mde.py \
    --checkpoint re-iqa_ckpts/mde_quality_aware.pth \
    --csv_path csv_files/koniq_test.csv \
    --output_path features/quality_mde.npy \
    --show_diagnosis   # optional: prints distortion diagnosis per image

# Content features (unchanged from original)
python demo_content_aware_feats.py \
    --head mlp
```

---

## Understanding the output

The gating weights give you a free distortion diagnosis per image.
With `--show_diagnosis`:

```
cat_sharp.jpg:    {'gaussian': 0.031, 'blur': 0.028, 'jpeg': 0.026, 'weather': 0.915}
sunset_blurry.jpg:{'gaussian': 0.041, 'blur': 0.872, 'jpeg': 0.052, 'weather': 0.035}
phone_photo.jpg:  {'gaussian': 0.621, 'blur': 0.152, 'jpeg': 0.191, 'weather': 0.036}
```

This tells you which distortion type dominates each image — a useful
by-product of the architecture that can be used for analysis or dataset auditing.

---

## Key design decisions

### Why a shared backbone?
Running four full ResNet-50s would quadruple memory and compute.
The backbone's early layers (edges, textures) are useful for all
distortion types equally, so sharing them loses nothing.

### Why ARNIQA manifolds instead of just four InfoNCE losses?
InfoNCE only teaches type discrimination — is this blurry or not?
The ARNIQA manifold additionally teaches severity ordering — is this
mildly blurry or heavily blurry? This severity signal is what gives
the quality score its fine-grained precision.

### Why soft MoE gating instead of hard assignment?
Real images often have multiple distortion types simultaneously
(a JPEG photo taken in fog). Soft weights handle mixed distortions
gracefully, while hard assignment would always miss half the signal.

### Why entropy regularisation for the gate?
Without it, the gating network collapses to always selecting the
head that converges first, preventing the others from specialising.
The entropy loss forces all four heads to remain active throughout training.

---

## Citation

If you use this work, please cite both original papers:

```bibtex
@InProceedings{Saha_2023_CVPR,
    author    = {Saha, Avinab and Mishra, Sandeep and Bovik, Alan C.},
    title     = {Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild},
    booktitle = {CVPR},
    year      = {2023},
}

@InProceedings{agnolucci2024arniqa,
    title     = {ARNIQA: Learning Distortion Manifold for Image Quality Assessment},
    author    = {Agnolucci, Lorenzo and Galteri, Leonardo and Bertini, Marco and Del Bimbo, Alberto},
    booktitle = {WACV},
    year      = {2024},
}
```
