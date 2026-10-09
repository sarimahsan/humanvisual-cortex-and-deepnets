"""
Feature Extraction Runner.

Extracts intermediate activations for a specified model or control baseline
across an image dataset, applying spatial pooling and saving float16 representations
to the disk cache.
"""

from typing import List, Optional
import argparse
import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from features.models import ModelWrapper
from features.gabor import GaborPyramidExtractor
from features.cache import FeatureCache
from data.algonauts import AlgonautsDataset


class ImageListDataset(Dataset):
    """Simple PyTorch Dataset for loading images from file paths."""

    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            if self.transform is not None:
                tensor = self.transform(rgb_img)
            else:
                tensor = transforms.ToTensor()(rgb_img)
        return tensor, idx


def extract_model_features(
    model_key: str,
    subject_id: str,
    data_root: str,
    config_path: str = "configs/models.yaml",
    cache_dir: str = "features/cache",
    batch_size: int = 64,
    device: Optional[str] = None,
) -> None:
    """Extracts features for all layers of model_key on subject_id's images."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = FeatureCache(cache_dir=cache_dir)
    with open(config_path, "r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)

    # 1. Load image list
    dataset = AlgonautsDataset(data_root=data_root, subject_id=subject_id)
    image_paths = dataset.get_image_paths()
    n_images = len(image_paths)
    print(f"[{model_key}] Extracting representations for {n_images} images (Subject {subject_id})...")

    # Handle Gabor pyramid baseline
    if model_key == "gabor_pyramid":
        if cache.exists(model_key, "multiscale_energy", subject_id):
            print(f"Cache already exists for {model_key} - Subject {subject_id}. Skipping.")
            return

        extractor = GaborPyramidExtractor(device=device)
        all_features = []
        for i in tqdm(range(0, n_images, batch_size), desc="Gabor extraction"):
            batch_paths = image_paths[i : i + batch_size]
            batch_feats = extractor.extract_batch(batch_paths)
            all_features.append(batch_feats)
        all_features_np = np.concatenate(all_features, axis=0)
        cache.save(
            all_features_np,
            model_key=model_key,
            layer_name="multiscale_energy",
            subject_id=subject_id,
            metadata={"normalized_depth": 0.05, "desc": "Gabor multiscale energy + color"},
        )
        print(f"Gabor features saved: shape {all_features_np.shape}")
        return

    # Handle Deep Neural Networks
    model_dict = models_cfg.get("models", {}).get(model_key)
    if not model_dict:
        raise ValueError(f"Model {model_key} not defined in {config_path}")

    # Check if all layers are already cached
    layers = model_dict.get("layers", [])
    needed_layers = [l for l in layers if not cache.exists(model_key, l["name"], subject_id)]
    if not needed_layers:
        print(f"All {len(layers)} layers already cached for {model_key} - {subject_id}. Skipping.")
        return

    wrapper = ModelWrapper(model_key, model_dict, device=device)
    loader = DataLoader(
        ImageListDataset(image_paths, transform=wrapper.transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if os.name != "nt" else 0,
    )

    layer_accumulators = {l["name"]: [] for l in layers}

    try:
        for batch_tensors, _ in tqdm(loader, desc=f"Forward passes [{model_key}]"):
            extracted = wrapper.extract_image_tensor(batch_tensors)
            for layer_name, act_np in extracted.items():
                layer_accumulators[layer_name].append(act_np)
    finally:
        wrapper.remove_hooks()

    # Save to disk
    for l_info in layers:
        l_name = l_info["name"]
        if layer_accumulators[l_name]:
            combined = np.concatenate(layer_accumulators[l_name], axis=0)
            cache.save(
                combined,
                model_key=model_key,
                layer_name=l_name,
                subject_id=subject_id,
                metadata={
                    "normalized_depth": l_info.get("normalized_depth", 0.5),
                    "desc": l_info.get("desc", ""),
                },
            )
            print(f"Saved {model_key} [{l_name}]: shape {combined.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Extraction Pipeline")
    parser.add_argument("--model", type=str, default="alexnet", help="Model key (alexnet, resnet50, etc.)")
    parser.add_argument("--subject", type=str, default="subj01", help="Subject identifier")
    parser.add_argument("--data_root", type=str, default="data/raw/algonauts_2023", help="Root data folder")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    extract_model_features(
        model_key=args.model,
        subject_id=args.subject,
        data_root=args.data_root,
        batch_size=args.batch_size,
        device=args.device,
    )
