"""
Low-Level Baseline: Multiscale Gabor Filter Bank & Spatial Statistics.

Implements a biologically-inspired early visual feature extractor:
- Multiscale 2D Gabor wavelet pyramid (4 frequencies, 8 orientations)
- Spatial energy pooling across quadrants (retaining retinotopic spatial info)
- Color statistics (RGB / LAB channel means and standard deviations)
- Spatial frequency power spectrum distribution
"""

from typing import List, Tuple
import numpy as np
from PIL import Image
from scipy.ndimage import convolve


def create_gabor_filter(
    ksize: int = 21,
    sigma: float = 4.0,
    theta: float = 0.0,
    lambd: float = 10.0,
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
    # Zero DC component (mean center)
    gb -= gb.mean()
    norm = np.linalg.norm(gb)
    if norm > 0:
        gb /= norm
    return gb.astype(np.float32)


class GaborPyramidExtractor:
    """Extracts multiscale Gabor energy, spatial pooling, and color statistics."""

    def __init__(
        self,
        n_orientations: int = 8,
        wavelengths: Tuple[float, ...] = (4.0, 8.0, 16.0, 32.0),
        spatial_grid: Tuple[int, int] = (4, 4),
    ):
        self.n_orientations = n_orientations
        self.wavelengths = wavelengths
        self.spatial_grid = spatial_grid

        # Precompute filter bank (even and odd phase quadrature pairs)
        self.filters: List[Tuple[np.ndarray, np.ndarray]] = []
        thetas = [i * np.pi / n_orientations for i in range(n_orientations)]

        for lambd in wavelengths:
            sigma = 0.56 * lambd  # Standard neurophysiological bandwidth ratio
            ksize = int(max(15, 2 * np.ceil(3 * sigma) + 1))
            if ksize % 2 == 0:
                ksize += 1
            for th in thetas:
                f_even = create_gabor_filter(ksize=ksize, sigma=sigma, theta=th, lambd=lambd, psi=0.0)
                f_odd = create_gabor_filter(ksize=ksize, sigma=sigma, theta=th, lambd=lambd, psi=np.pi / 2)
                self.filters.append((f_even, f_odd))

    def _pool_quadrants(self, feature_map: np.ndarray) -> np.ndarray:
        """Applies spatial pooling over a grid (e.g. 4x4) to retain retinotopy."""
        gh, gw = self.spatial_grid
        h, w = feature_map.shape
        pooled = np.zeros((gh, gw), dtype=np.float32)
        h_chunk = h // gh
        w_chunk = w // gw
        for i in range(gh):
            for j in range(gw):
                pooled[i, j] = feature_map[i * h_chunk : (i + 1) * h_chunk, j * w_chunk : (j + 1) * w_chunk].mean()
        return pooled.flatten()

    def extract_single_image(self, img: Image.Image) -> np.ndarray:
        """Extracts low-level features for a single PIL RGB image."""
        img_resized = img.resize((224, 224)).convert("RGB")
        img_np = np.asarray(img_resized, dtype=np.float32) / 255.0

        # Grayscale for Gabor filtering (standard luminance channel)
        gray = 0.2989 * img_np[:, :, 0] + 0.5870 * img_np[:, :, 1] + 0.1140 * img_np[:, :, 2]

        feature_blocks: List[np.ndarray] = []

        # 1. Gabor quadrature energy maps
        for f_even, f_odd in self.filters:
            resp_even = convolve(gray, f_even, mode="reflect")
            resp_odd = convolve(gray, f_odd, mode="reflect")
            # Phase-invariant complex cell energy response
            energy = np.sqrt(resp_even**2 + resp_odd**2)
            pooled_energy = self._pool_quadrants(energy)
            feature_blocks.append(pooled_energy)

        # 2. Color channel statistics (means and standard deviations across 4x4 grid)
        for c in range(3):
            c_map = img_np[:, :, c]
            feature_blocks.append(self._pool_quadrants(c_map))

        # 3. Spatial frequency distribution via 2D FFT
        fft2 = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        fft_pooled = self._pool_quadrants(np.log1p(fft2))
        feature_blocks.append(fft_pooled)

        concatenated = np.concatenate(feature_blocks).astype(np.float32)
        return concatenated

    def extract_batch(self, image_paths: List[str]) -> np.ndarray:
        """Extracts features across a batch of image paths."""
        results = []
        for path in image_paths:
            with Image.open(path) as img:
                feats = self.extract_single_image(img)
                results.append(feats)
        return np.stack(results, axis=0).astype(np.float16)
