"""
Unified Deep Neural Network Feature Extractor.

Supports:
- AlexNet (Supervised ImageNet)
- ResNet-50 (Supervised ImageNet)
- ResNet-50 Untrained (Random initialization control)
- DeiT-S / ViT-S/16 (Supervised ImageNet)
- DINO ViT-S/16 (Self-Supervised)
- CLIP ViT-B/16 (Language-Image Contrastive)

Applies spatial pooling (2x2 / 4x4 adaptive pooling for CNNs, CLS + pooled patch tokens for ViTs)
to preserve retinotopic position for early visual cortex representations.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
from torchvision import transforms
from PIL import Image
import numpy as np


class SpatialFeatureHook:
    """Captures intermediate activations during forward pass and applies spatial pooling."""

    def __init__(self, pooling_config: Dict[str, Any], is_vit: bool = False):
        self.pooling_config = pooling_config
        self.is_vit = is_vit
        self.activation: Optional[torch.Tensor] = None

    def __call__(self, module: nn.Module, input_tensor: Any, output_tensor: Any):
        # Handle cases where output is a tuple (e.g. transformers or RNNs)
        if isinstance(output_tensor, tuple):
            act = output_tensor[0]
        else:
            act = output_tensor

        with torch.no_grad():
            if self.is_vit:
                # ViT tensor: (B, 1 + N_patches, D) or (B, N_patches, D)
                if act.ndim == 3:
                    b, seq_len, dim = act.shape
                    if seq_len == 197:  # 1 CLS + 14x14 patches
                        cls_tok = act[:, 0, :]  # (B, D)
                        patch_toks = act[:, 1:, :]  # (B, 196, D)
                        # Reshape patches to spatial grid (B, D, 14, 14)
                        patch_grid = patch_toks.transpose(1, 2).reshape(b, dim, 14, 14)
                        # Adaptive pool patch grid to 2x2
                        grid_sz = tuple(self.pooling_config.get("patch_grid", [2, 2]))
                        pooled_patches = F.adaptive_avg_pool2d(patch_grid, grid_sz)  # (B, D, 2, 2)
                        pooled_flat = pooled_patches.flatten(1)  # (B, D * 4)
                        combined = torch.cat([cls_tok, pooled_flat], dim=1)  # (B, 5 * D)
                        self.activation = combined.detach().cpu()
                    else:
                        self.activation = act.flatten(1).detach().cpu()
                else:
                    self.activation = act.flatten(1).detach().cpu()
            else:
                # CNN 4D tensor: (B, C, H, W)
                if act.ndim == 4:
                    out_sz = tuple(self.pooling_config.get("output_size", [2, 2]))
                    pooled = F.adaptive_avg_pool2d(act, out_sz)
                    self.activation = pooled.flatten(1).detach().cpu()
                else:
                    # Fully connected layer (B, D)
                    self.activation = act.flatten(1).detach().cpu()


class ModelWrapper:
    """Wrapper that loads public models and extracts representations at specified layer taps."""

    def __init__(self, model_key: str, model_cfg: Dict[str, Any], device: str = "cuda"):
        self.model_key = model_key
        self.model_cfg = model_cfg
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model, self.transform = self._load_model_and_transform()
        self.model.eval()
        self.model.to(self.device)

        self.hooks: Dict[str, SpatialFeatureHook] = {}
        self.hook_handles: List[Any] = []
        self._register_hooks()

    def _load_model_and_transform(self) -> Tuple[nn.Module, Any]:
        m_type = self.model_cfg.get("type", "cnn")
        source = self.model_cfg.get("source", "torchvision")

        # Standard ImageNet normalization
        standard_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.model_key == "alexnet":
            weights = tv_models.AlexNet_Weights.DEFAULT if self.model_cfg.get("weights") else None
            model = tv_models.alexnet(weights=weights)
            return model, standard_transform

        elif self.model_key == "resnet50":
            weights = tv_models.ResNet50_Weights.DEFAULT if self.model_cfg.get("weights") else None
            model = tv_models.resnet50(weights=weights)
            return model, standard_transform

        elif self.model_key == "resnet50_untrained":
            model = tv_models.resnet50(weights=None)
            return model, standard_transform

        elif self.model_key in ["deit_small", "dino_vit_small"]:
            import timm
            m_name = self.model_cfg.get("model_name", "deit_small_patch16_224")
            pretrained = self.model_cfg.get("pretrained", True)
            try:
                model = timm.create_model(m_name, pretrained=pretrained)
            except Exception:
                # Fallback to standard deit or vit
                fallback_name = "deit_small_patch16_224" if "deit" in m_name else "vit_small_patch16_224"
                model = timm.create_model(fallback_name, pretrained=True)
            return model, standard_transform

        elif self.model_key == "clip_vit_b16":
            import open_clip
            model_name = self.model_cfg.get("model_name", "ViT-B-16")
            pretrained = self.model_cfg.get("pretrained", "openai")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            # Use visual branch of CLIP
            visual_model = model.visual
            return visual_model, preprocess

        else:
            raise ValueError(f"Unknown model_key: {self.model_key}")

    def _register_hooks(self) -> None:
        pooling_cfg = self.model_cfg.get("pooling", {})
        is_vit = self.model_cfg.get("type") in ["vit", "clip"]

        named_modules = dict(self.model.named_modules())
        for layer_info in self.model_cfg.get("layers", []):
            layer_name = layer_info["name"]
            if layer_name in named_modules:
                hook = SpatialFeatureHook(pooling_cfg, is_vit=is_vit)
                handle = named_modules[layer_name].register_forward_hook(hook)
                self.hooks[layer_name] = hook
                self.hook_handles.append(handle)
            else:
                print(f"Warning: layer '{layer_name}' not found in {self.model_key}. Available modules: {list(named_modules.keys())[:10]}...")

    def extract_image_tensor(self, batch_tensors: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Runs forward pass on batch of preprocessed image tensors (B, 3, 224, 224)
        and returns dictionary of {layer_name: numpy_array_shape_(B, D)}.
        """
        batch_tensors = batch_tensors.to(self.device)
        with torch.no_grad():
            try:
                _ = self.model(batch_tensors)
            except Exception as e:
                # Some transformer blocks expect special args
                raise RuntimeError(f"Error during forward pass for {self.model_key}: {e}")

        extracted = {}
        for layer_name, hook in self.hooks.items():
            if hook.activation is not None:
                extracted[layer_name] = hook.activation.numpy().astype(np.float16)
        return extracted

    def remove_hooks(self) -> None:
        """Cleans up registered forward hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.hooks.clear()
