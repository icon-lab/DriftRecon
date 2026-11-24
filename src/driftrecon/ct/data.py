"""CT slice datasets for DriftReconCT."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _normalize_slice(
    array: np.ndarray,
    window: Optional[tuple[float, float]],
) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("CT slice contains NaN or infinite values.")
    if window is not None:
        low, high = float(window[0]), float(window[1])
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("window must contain finite increasing values.")
        return np.clip((array - low) / (high - low), 0.0, 1.0)
    minimum, maximum = float(array.min()), float(array.max())
    if minimum >= 0.0 and maximum <= 1.0:
        return array
    if maximum <= minimum:
        return np.zeros_like(array)
    return (array - minimum) / (maximum - minimum)


def load_ct_slice(
    path: str | Path,
    *,
    array_key: str = "image",
    window: Optional[tuple[float, float]] = None,
    image_size: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Load one 2-D CT slice from ``.npy``, ``.npz`` or an optional image file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if array_key in archive:
                array = archive[array_key]
            elif len(archive.files) == 1:
                array = archive[archive.files[0]]
            else:
                raise KeyError(
                    f"{path} has keys {archive.files}; pass the intended array_key."
                )
    elif suffix in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
        try:
            from PIL import Image
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Install driftrecon[images] to load image files.") from error
        array = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported CT slice extension: {suffix!r}.")
    array = np.asarray(array).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected one 2-D slice in {path}, got shape {array.shape}.")
    tensor = torch.from_numpy(_normalize_slice(array, window)).unsqueeze(0)
    if image_size is not None and tuple(tensor.shape[-2:]) != tuple(image_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=tuple(int(value) for value in image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor.contiguous()


def load_ct_sinogram(
    path: str | Path,
    *,
    array_key: str = "sinogram",
    expected_shape: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """Load one measured sinogram as ``(1, 1, views, detectors)``."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if array_key in archive:
                array = archive[array_key]
            elif len(archive.files) == 1:
                array = archive[archive.files[0]]
            else:
                raise KeyError(
                    f"{path} has keys {archive.files}; pass the intended sinogram key."
                )
    else:
        raise ValueError("Measured sinograms must be stored as .npy or .npz.")
    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(
            f"Expected one (views, detectors) sinogram, got shape {array.shape}."
        )
    if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
        raise ValueError(
            f"Expected sinogram shape {expected_shape}, got {tuple(array.shape)}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Sinogram contains NaN or infinite values.")
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).contiguous()


class CTSliceDataset(Dataset[torch.Tensor]):
    """Directory of independent 2-D CT slices.

    Clinical HU arrays should pass an explicit ``window``.  Pre-normalized
    research slices in ``[0,1]`` are preserved as-is.  ``augment`` applies
    independent random horizontal and vertical flips (training only).
    """

    EXTENSIONS = {".npy", ".npz", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}

    def __init__(
        self,
        root: str | Path,
        *,
        image_size: tuple[int, int],
        window: Optional[tuple[float, float]] = None,
        array_key: str = "image",
        max_slices: Optional[int] = None,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.augment = bool(augment)
        if not self.root.is_dir():
            raise FileNotFoundError(f"CT data directory does not exist: {self.root}")
        self.image_size = tuple(int(value) for value in image_size)
        self.window = window
        self.array_key = str(array_key)
        paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in self.EXTENSIONS
        )
        if max_slices is not None:
            paths = paths[: int(max_slices)]
        if not paths:
            raise FileNotFoundError(f"No supported CT slices found under {self.root}.")
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = load_ct_slice(
            self.paths[index],
            array_key=self.array_key,
            window=self.window,
            image_size=self.image_size,
        )
        if self.augment:
            if torch.rand(()) < 0.5:
                image = image.flip(-1)
            if torch.rand(()) < 0.5:
                image = image.flip(-2)
            image = image.contiguous()
        return image


def make_ellipse_phantom(index: int, image_size: int) -> torch.Tensor:
    """Create a deterministic CT-like phantom with varied ellipses."""
    if int(image_size) <= 0:
        raise ValueError("image_size must be positive.")
    coordinates = torch.linspace(-1.0, 1.0, int(image_size))
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    image = torch.zeros_like(xx)
    ellipses = [
        (0.0, 0.0, 0.72, 0.90, 0.65),
        (-0.22, -0.18, 0.22, 0.34, 0.22),
        (0.28, 0.12, 0.18, 0.28, 0.38),
        (0.02, 0.42, 0.30, 0.12, 0.30),
    ]
    shift = 0.025 * ((int(index) % 5) - 2)
    for ellipse_index, (cx, cy, rx, ry, value) in enumerate(ellipses):
        angle = 0.13 * (int(index) + ellipse_index)
        cosine, sine = math.cos(angle), math.sin(angle)
        x_centered, y_centered = xx - cx - shift, yy - cy + shift
        x_rotated = cosine * x_centered + sine * y_centered
        y_rotated = -sine * x_centered + cosine * y_centered
        mask = (x_rotated / rx).square() + (y_rotated / ry).square() <= 1.0
        image = image + float(value) * mask
    # A small low-density insert forces the handful of phantoms to differ in
    # both positive and negative contrast while remaining normalized.
    hole_x = -0.35 + 0.12 * (int(index) % 4)
    hole = (xx - hole_x).square() + (yy + 0.28).square() < 0.08**2
    image = image - 0.2 * hole
    image = image.clamp_min(0.0)
    return (image / image.max().clamp_min(1e-8)).unsqueeze(0)


class SyntheticPhantomDataset(Dataset[torch.Tensor]):
    """Deterministic phantoms for tests and installation smoke checks."""

    def __init__(self, count: int = 8, image_size: int = 64) -> None:
        if int(count) <= 0:
            raise ValueError("count must be positive.")
        self.count = int(count)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> torch.Tensor:
        return make_ellipse_phantom(index, self.image_size)
