"""
Synthetic Algonauts / NSD Dataset Generator for Local Testing and Validation.

Creates small, lightweight synthetic datasets matching Algonauts 2023 folder structure
and shapes to allow instant testing and verification of the full pipeline.
"""

from typing import Tuple
import os
import numpy as np
from PIL import Image

from data.rois import ROI_PALETTE


def generate_synthetic_dataset(
    output_dir: str = "data/raw/synthetic_demo",
    subject_id: str = "subj01",
    n_images: int = 200,
    n_lh_vertices: int = 500,
    n_rh_vertices: int = 500,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Generates synthetic subject data matching the Algonauts 2023 format.
    Returns:
        (images_dir, fmri_dir)
    """
    rng = np.random.RandomState(seed)

    subj_dir = os.path.join(output_dir, subject_id)
    images_dir = os.path.join(subj_dir, "training_split", "training_images")
    fmri_dir = os.path.join(subj_dir, "training_split", "training_fmri")
    roi_dir = os.path.join(subj_dir, "roi_masks")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(fmri_dir, exist_ok=True)
    os.makedirs(roi_dir, exist_ok=True)

    # 1. Create synthetic images (diverse colored patches with edges and gradients)
    print(f"Generating {n_images} synthetic images in {images_dir}...")
    for i in range(n_images):
        arr = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        # Add some structure: gradient or circular patch
        if i % 3 == 0:
            arr[:, :, 0] = np.linspace(0, 255, 224, dtype=np.uint8)[:, None]
        elif i % 3 == 1:
            arr[:, :, 1] = np.linspace(255, 0, 224, dtype=np.uint8)[None, :]
        img = Image.fromarray(arr)
        img.save(os.path.join(images_dir, f"train_{i:04d}.png"))

    # 2. Create ROI masks
    # Divide vertices into realistic ROI partitions
    rois = ["V1", "V2", "V3", "hV4", "FFA", "PPA", "EBA", "OFA"]
    for hemi, n_v in [("lh", n_lh_vertices), ("rh", n_rh_vertices)]:
        chunk = n_v // (len(rois) + 1)
        for idx, r in enumerate(rois):
            mask = np.zeros(n_v, dtype=bool)
            start = idx * chunk
            end = (idx + 1) * chunk
            mask[start:end] = True
            np.save(os.path.join(roi_dir, f"{hemi}.{r}.npy"), mask)

    # 3. Create synthetic fMRI responses with mild hierarchical correlations
    # (z-scored responses per vertex)
    print(f"Generating synthetic fMRI responses for {n_lh_vertices} LH and {n_rh_vertices} RH vertices...")
    lh_fmri = rng.randn(n_images, n_lh_vertices).astype(np.float32)
    rh_fmri = rng.randn(n_images, n_rh_vertices).astype(np.float32)

    # Inject slight signal into vertices
    latent_signal = rng.randn(n_images, 1)
    lh_fmri[:, :100] += 0.5 * latent_signal
    rh_fmri[:, :100] += 0.5 * latent_signal

    np.save(os.path.join(fmri_dir, "lh_training_fmri.npy"), lh_fmri)
    np.save(os.path.join(fmri_dir, "rh_training_fmri.npy"), rh_fmri)

    print(f"Synthetic dataset created successfully at {subj_dir}")
    return images_dir, fmri_dir
