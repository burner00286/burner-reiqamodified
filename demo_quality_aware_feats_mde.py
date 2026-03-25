"""
demo_quality_aware_feats_mde.py
================================
Extract quality-aware features using the trained MultiDistortionEncoder.

This is the equivalent of Re-IQA's demo_quality_aware_feats.py,
updated to use our new encoder. Everything downstream (concatenating
with content features, training the regressor) stays the same.

WHAT THIS SCRIPT DOES:
-----------------------
1. Loads a trained MultiDistortionEncoder checkpoint
2. Runs a set of images through it
3. Saves the 128-dim quality embeddings to a .npy file
4. Optionally prints the gating weights (distortion diagnosis)

USAGE:
------
    # Extract quality features
    python demo_quality_aware_feats_mde.py \
        --checkpoint re-iqa_ckpts/mde_quality_aware.pth \
        --csv_path csv_files/test_images.csv \
        --output_path features/quality_feats_mde.npy

    # Also show distortion diagnosis per image
    python demo_quality_aware_feats_mde.py \
        --checkpoint re-iqa_ckpts/mde_quality_aware.pth \
        --csv_path csv_files/test_images.csv \
        --output_path features/quality_feats_mde.npy \
        --show_diagnosis

THEN (same as original Re-IQA):
    Concatenate with content features and train regressor:
        quality = np.load('features/quality_feats_mde.npy')
        content = np.load('features/content_feats.npy')
        combined = np.concatenate([quality, content], axis=1)
        # Train Ridge regressor on combined features
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms as T

from networks.multi_distortion_encoder import MultiDistortionEncoder


# ─────────────────────────────────────────────────────────────
#  IMAGE PREPROCESSING
#  Standard ImageNet normalisation — same as original Re-IQA
# ─────────────────────────────────────────────────────────────

PREPROCESS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path: str, device: torch.device) -> MultiDistortionEncoder:
    """
    Load a trained MultiDistortionEncoder from a checkpoint.

    Args:
        checkpoint_path : path to .pth checkpoint file
        device          : torch device to load onto

    Returns:
        model in eval mode, moved to device
    """
    model = MultiDistortionEncoder()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both plain state_dict and wrapped checkpoints
    if 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
        # Strip 'module.encoder_q.' prefix if saved from DDP/MoCo
        state = {
            k.replace('module.encoder_q.', '').replace('encoder_q.', ''): v
            for k, v in state.items()
            if 'encoder_q' in k
        }
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f'Loaded checkpoint from: {checkpoint_path}')
    return model


@torch.no_grad()
def extract_features(
    model: MultiDistortionEncoder,
    image_paths: list,
    device: torch.device,
    batch_size: int = 32,
    show_diagnosis: bool = False,
) -> np.ndarray:
    """
    Extract quality embeddings for a list of images.

    Args:
        model        : trained MultiDistortionEncoder
        image_paths  : list of image file paths
        device       : torch device
        batch_size   : images to process at once (adjust for your VRAM)
        show_diagnosis: if True, prints gating weights per image

    Returns:
        features: (N, 128) numpy array of quality embeddings
    """
    all_features = []
    distortion_types = model.DISTORTION_TYPES

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                batch_tensors.append(PREPROCESS(img))
            except Exception as e:
                print(f'Warning: could not load {path}: {e}')
                # Insert zeros for failed images
                batch_tensors.append(torch.zeros(3, 224, 224))

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)

        # Get fused embedding + gating weights
        embeddings, weights = model.forward_with_weights(batch)

        all_features.append(embeddings.cpu().numpy())

        # Optional: print distortion diagnosis
        if show_diagnosis:
            w = weights.cpu().numpy()
            for j, path in enumerate(batch_paths):
                diag = {
                    dtype: f'{w[j, k]:.3f}'
                    for k, dtype in enumerate(distortion_types)
                }
                print(f'{Path(path).name}: {diag}')

        if (i // batch_size) % 10 == 0:
            print(f'Processed {min(i + batch_size, len(image_paths))}'
                  f'/{len(image_paths)} images')

    return np.vstack(all_features)   # (N, 128)


def main():
    parser = argparse.ArgumentParser('Extract MDE quality features')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained MDE checkpoint (.pth)')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='CSV with image paths (same format as Re-IQA)')
    parser.add_argument('--output_path', type=str,
                        default='features/quality_feats_mde.npy',
                        help='Where to save the extracted features')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--show_diagnosis', action='store_true',
                        help='Print distortion type weights per image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    model = load_model(args.checkpoint, device)

    # Load image paths from CSV (same format as Re-IQA)
    import pandas as pd
    df = pd.read_csv(args.csv_path)
    # Re-IQA's CSV has image paths in the first column
    image_paths = df.iloc[:, 0].tolist()
    print(f'Extracting features for {len(image_paths)} images...')

    # Extract features
    features = extract_features(
        model, image_paths, device,
        batch_size=args.batch_size,
        show_diagnosis=args.show_diagnosis,
    )

    # Save
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, features)
    print(f'Saved features: {features.shape} -> {args.output_path}')
    print()
    print('Next step: concatenate with content features and train regressor.')
    print('  quality = np.load("features/quality_feats_mde.npy")')
    print('  content = np.load("features/content_feats.npy")')
    print('  combined = np.concatenate([quality, content], axis=1)')


if __name__ == '__main__':
    main()
