"""
Low-Level Baseline: Multiscale Gabor Filter Bank & Spatial Statistics (High-Speed Vectorized).

Implements biologically-inspired early visual feature extraction:
- Multiscale 2D Gabor wavelet pyramid (4 frequencies, 8 orientations)
- Accelerated via PyTorch Convolutions with efficient kernel sizing
- Spatial energy pooling across quadrants (retaining retinotopic spatial info)
- Color statistics across quadrants
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def create_gabor_filter(
    ksize: int = 15,
    sigma: float = 2.5,
    theta: float = 0.0,
    lambd: float = 5.0,
    gamma: float = 0.5,
    psi: float = 0.0,
) -> np.ndarray:
    """Generates a 2D Gabor wavelet kernel."""
    sigma_x = sigma
    sigma_y = sigma / gamma

    half = ksize // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1]

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * np.cos(
        2 * np.pi * x_theta / lambd + psi
    )
    gb -= gb.mean()
    norm = np.linalg.norm(gb)
    if norm > 0:
        gb /= norm
    return gb.astype(np.float32)


class GaborPyramidExtractor:
    """High-speed Gabor energy and color statistics extractor."""

    def __init__(
        self,
        n_orientations: int = 8,
        wavelengths: Tuple[float, ...] = (3.0, 5.0, 8.0, 12.0),
        spatial_grid: Tuple[int, int] = (4, 4),
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.spatial_grid = spatial_grid

        # 15x15 kernels
        max_ksize = 15
        thetas = [i * np.pi / n_orientations for i in range(n_orientations)]

        even_filters = []
        odd_filters = []

        for lambd in wavelengths:
            sigma = 0.56 * lambd
            for th in thetas:
                f_even = create_gabor_filter(ksize=max_ksize, sigma=sigma, theta=th, lambd=lambd, psi=0.0)
                f_odd = create_gabor_filter(ksize=max_ksize, sigma=sigma, theta=th, lambd=lambd, psi=np.pi / 2)
                even_filters.append(f_even)
                odd_filters.append(f_odd)

        even_t = torch.tensor(np.stack(even_filters)[:, None, :, :], dtype=torch.float32, device=self.device)
        odd_t = torch.tensor(np.stack(odd_filters)[:, None, :, :], dtype=torch.float32, device=self.device)

        self.even_weight = even_t
        self.odd_weight = odd_t
        self.pad = max_ksize // 2

    def extract_batch(self, image_paths: List[str]) -> np.ndarray:
        """Extracts features across a batch of images quickly."""
        tensors = []
        for p in image_paths:
            with Image.open(p) as img:
                # Downsample to 112x112 for high-speed spatial filtering
                im = img.convert("RGB").resize((112, 112))
                arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
                tensors.append(arr)

        batch = torch.tensor(np.stack(tensors), dtype=torch.float32, device=self.device)

        # Luminance
        gray = 0.2989 * batch[:, 0:1] + 0.5870 * batch[:, 1:2] + 0.1140 * batch[:, 2:3]

        with torch.no_grad():
            resp_even = F.conv2d(gray, self.even_weight, padding=self.pad)
            resp_odd = F.conv2d(gray, self.odd_weight, padding=self.pad)
            energy = torch.sqrt(resp_even**2 + resp_odd**2 + 1e-8)

            pooled_energy = F.adaptive_avg_pool2d(energy, self.spatial_grid).flatten(1)
            pooled_color = F.adaptive_avg_pool2d(batch, self.spatial_grid).flatten(1)

            features = torch.cat([pooled_energy, pooled_color], dim=1)

        return features.cpu().numpy().astype(np.float16)
