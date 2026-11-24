"""M4Raw multi-coil data for DriftReconMRI.

Protocol
--------
* The first ``slices_per_volume`` slices of every ``multicoil_<phase>/*.h5``
  file are used (fully sampled, first repetition).
* Coil maps are estimated once per slice with ESPIRiT (sigpy) from the central
  calibration region and cached together with the raw k-space.
* Every volume is normalized by the peak magnitude of its fully sampled
  SENSE-adjoint image (orthonormal FFT).
* Undersampling uses 2-D Gaussian variable-density masks with a fully sampled
  8% center; masks are deterministic per slice and acceleration.

Each sample provides ``x_fs`` (reference), ``x_us = A^H y`` and the measured
k-space ``y_us``, together with the mask, coil maps and the contrast label.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

CONTRAST_TO_ID = {"T1": 0, "T2": 1, "FLAIR": 2}
ID_TO_CONTRAST = {value: key for key, value in CONTRAST_TO_ID.items()}
CONTRASTS = tuple(CONTRAST_TO_ID)

ACS_CENTER_FRACTION = 0.08
ESPIRIT_CROP = 0.8
ESPIRIT_KERNEL_WIDTH = 6
ESPIRIT_THRESHOLD = 0.02
ESPIRIT_DEVICE = os.environ.get("M4RAW_ESPIRIT_DEVICE", "auto")

KSPACE_CACHE_VERSION = "driftrecon_m4raw_kspace_espirit_crop08_v1"
VOLUME_SCALES_CACHE_VERSION = "driftrecon_m4raw_volume_scales_v1"


# --------------------------------------------------------------------------
# Centered orthonormal FFTs and SENSE operators (NumPy)
# --------------------------------------------------------------------------
def kspace_to_coil_images(kspace: np.ndarray) -> np.ndarray:
    """Centered orthonormal 2-D IFFT of ``(..., coils, H, W)`` k-space."""
    kspace = np.asarray(kspace, dtype=np.complex64)
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(kspace, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    ).astype(np.complex64, copy=False)


def sense_combine(coil_images: np.ndarray, coil_maps: np.ndarray) -> np.ndarray:
    """SENSE adjoint coil combination ``sum_c conj(S_c) * x_c``."""
    return np.sum(np.conj(coil_maps) * coil_images, axis=-3).astype(np.complex64, copy=False)


def sense_adjoint(kspace: np.ndarray, coil_maps: np.ndarray) -> np.ndarray:
    """Coil-combined complex image ``A^H y`` for one ``(coils, H, W)`` slice."""
    return sense_combine(kspace_to_coil_images(kspace), coil_maps)


def complex_to_2ch(z: np.ndarray) -> np.ndarray:
    return np.stack([np.real(z), np.imag(z)], axis=0).astype(np.float32, copy=False)


def kspace_to_2ch(kspace: np.ndarray) -> torch.Tensor:
    """``(coils, H, W)`` complex k-space to ``(2, coils, H, W)`` real/imag."""
    tensor = torch.as_tensor(np.asarray(kspace, dtype=np.complex64))
    return torch.view_as_real(tensor).permute(3, 0, 1, 2).contiguous()


# --------------------------------------------------------------------------
# Undersampling masks
# --------------------------------------------------------------------------
def acs_region_bounds(height: int, width: int, center_fraction: float = ACS_CENTER_FRACTION):
    center_h = int(height * center_fraction)
    center_w = int(width * center_fraction)
    return height // 2 - center_h // 2, width // 2 - center_w // 2, center_h, center_w


def parse_r_values(value: Union[str, Sequence[float]]) -> Tuple[float, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    if not parts:
        raise ValueError("At least one acceleration factor is required.")
    return tuple(float(part) for part in parts)


def gaussian_mask(height: int, width: int, R: float, seed: Optional[int] = None) -> np.ndarray:
    """2-D Gaussian variable-density mask ``(H, W)`` with a fully sampled center."""
    total_pixels = height * width
    if R <= 1.0:
        return np.ones((height, width), dtype=np.float32)
    num_sampled = int(total_pixels / R)
    center_h = int(height * ACS_CENTER_FRACTION)
    center_w = int(width * ACS_CENTER_FRACTION)
    if center_h * center_w > num_sampled:
        center_h = int(np.sqrt(num_sampled))
        center_w = int(np.sqrt(num_sampled))
    start_h = height // 2 - center_h // 2
    start_w = width // 2 - center_w // 2

    y, x = np.indices((height, width))
    y = y - height // 2
    x = x - width // 2
    std_scale = 0.5 / np.sqrt(R)
    std_y, std_x = height * std_scale, width * std_scale
    pdf = np.exp(-(x**2 / (2 * std_x**2) + y**2 / (2 * std_y**2))) + 1e-8
    pdf[start_h : start_h + center_h, start_w : start_w + center_w] = 0.0
    pdf_flat = pdf.flatten()
    pdf_flat /= pdf_flat.sum()
    num_remaining = num_sampled - center_h * center_w

    # The legacy global NumPy stream is used (and restored) so that masks are
    # identical to the ones used for the reported experiments.
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    mask_flat = np.zeros(total_pixels, dtype=np.float32)
    if num_remaining > 0:
        indices = np.random.choice(total_pixels, size=num_remaining, replace=False, p=pdf_flat)
        mask_flat[indices] = 1.0
    mask = mask_flat.reshape(height, width)
    mask[start_h : start_h + center_h, start_w : start_w + center_w] = 1.0
    if seed is not None:
        np.random.set_state(state)
    return mask.astype(np.float32, copy=False)


# --------------------------------------------------------------------------
# ESPIRiT coil maps (sigpy)
# --------------------------------------------------------------------------
def resolve_espirit_device(device: Optional[Union[str, int]] = None):
    """Return ``(sigpy device, name)``; ``auto`` uses the first GPU when cupy works."""
    import sigpy as sp

    if device is None:
        device = ESPIRIT_DEVICE
    if isinstance(device, str):
        name = device.strip().lower()
        if name in {"cpu", "none", "false"}:
            return sp.cpu_device, "cpu"
        if name in {"auto", ""}:
            try:
                import cupy

                if cupy.cuda.runtime.getDeviceCount() > 0:
                    return sp.Device(0), "cuda:0"
            except Exception:
                pass
            return sp.cpu_device, "cpu"
        if name.startswith("cuda:"):
            name = name.split(":", 1)[1]
        if not name.isdigit():
            raise ValueError(f"Unsupported ESPIRiT device {device!r}; use auto, cpu, 0 or cuda:N.")
        device = int(name)
    if isinstance(device, int):
        try:
            return sp.Device(int(device)), f"cuda:{int(device)}"
        except ValueError as exc:
            raise ValueError(
                f"ESPIRiT GPU device {device} unavailable; install cupy for your CUDA version."
            ) from exc
    raise TypeError(f"Unsupported ESPIRiT device type: {type(device)}")


def espirit_coil_maps(
    kspace: np.ndarray,
    *,
    crop: float = ESPIRIT_CROP,
    device: Optional[Union[str, int]] = None,
) -> np.ndarray:
    """ESPIRiT maps for ``(coils, H, W)`` or ``(N, coils, H, W)`` fully sampled k-space."""
    import sigpy as sp
    import sigpy.mri as mr

    sigpy_device, _ = resolve_espirit_device(device)

    def run_slice(kspace_slice: np.ndarray) -> np.ndarray:
        kspace_slice = np.asarray(kspace_slice, dtype=np.complex64)
        _, _, center_h, center_w = acs_region_bounds(*kspace_slice.shape[-2:])
        app = mr.app.EspiritCalib(
            sp.to_device(kspace_slice, sigpy_device),
            calib_width=int(max(center_h, center_w)),
            kernel_width=ESPIRIT_KERNEL_WIDTH,
            thresh=ESPIRIT_THRESHOLD,
            crop=float(crop),
            show_pbar=False,
            device=sigpy_device,
        )
        return np.asarray(sp.to_device(app.run(), sp.cpu_device), dtype=np.complex64)

    kspace = np.asarray(kspace, dtype=np.complex64)
    if kspace.ndim == 3:
        return run_slice(kspace)
    if kspace.ndim == 4:
        return np.stack([run_slice(kspace[index]) for index in range(kspace.shape[0])])
    raise ValueError(f"Expected k-space (C,H,W) or (N,C,H,W), got {kspace.shape}.")


# --------------------------------------------------------------------------
# Disk cache: raw k-space, coil maps and per-volume scales
# --------------------------------------------------------------------------
def phase_folder(data_root: str, phase: str) -> str:
    return os.path.join(data_root, f"multicoil_{phase}")


def list_volumes(data_root: str, phase: str, contrast: str, max_volumes: Optional[int]) -> List[str]:
    """Sorted ``.h5`` names of one split, optionally filtered by contrast."""
    folder = phase_folder(data_root, phase)
    names = sorted(
        name
        for name in os.listdir(folder)
        if name.endswith(".h5")
        and (contrast == "all" or name[11 : 11 + len(contrast)] == contrast)
    )
    if max_volumes is not None:
        names = names[: int(max_volumes)]
    return names


def _cache_digest(**key) -> str:
    key = dict(key, data_root=os.path.abspath(key["data_root"]), version=KSPACE_CACHE_VERSION)
    return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def _cache_paths(cache_dir: str, digest: str) -> Dict[str, str]:
    return {
        "kspace": os.path.join(cache_dir, f"m4raw_kspace_{digest}.npy"),
        "coil_maps": os.path.join(cache_dir, f"m4raw_coilmaps_{digest}.npy"),
        "meta": os.path.join(cache_dir, f"m4raw_kspace_{digest}.meta.json"),
        "scales": os.path.join(cache_dir, f"m4raw_volume_scales_{digest}.npy"),
        "scales_meta": os.path.join(cache_dir, f"m4raw_volume_scales_{digest}.meta.json"),
    }


def _build_kspace_cache(
    paths: Dict[str, str],
    *,
    data_root: str,
    phase: str,
    contrast: str,
    slices_per_volume: int,
    max_volumes: Optional[int],
    espirit_device,
    espirit_crop: float,
) -> None:
    import h5py
    from tqdm import tqdm

    _, device_name = resolve_espirit_device(espirit_device)
    folder = phase_folder(data_root, phase)
    names = list_volumes(data_root, phase, contrast, max_volumes)
    if not names:
        raise FileNotFoundError(f"No M4Raw files found under {folder}.")
    with h5py.File(os.path.join(folder, names[0]), "r") as handle:
        _, n_coils, height, width = handle["kspace"].shape
    num_slices = len(names) * slices_per_volume
    print(f"Building M4Raw cache ({phase}): {len(names)} volumes, {num_slices} slices, "
          f"ESPIRiT on {device_name} -> {paths['kspace']}")
    shape = (num_slices, int(n_coils), int(height), int(width))
    kspace_mm = np.lib.format.open_memmap(paths["kspace"], mode="w+", dtype=np.complex64, shape=shape)
    maps_mm = np.lib.format.open_memmap(paths["coil_maps"], mode="w+", dtype=np.complex64, shape=shape)
    slice_ids: List[str] = []
    offset = 0
    for name in tqdm(names, desc=f"M4Raw cache ({phase})"):
        with h5py.File(os.path.join(folder, name), "r") as handle:
            kspace = handle["kspace"][:slices_per_volume].astype(np.complex64)
        if kspace.shape[0] != slices_per_volume:
            raise ValueError(f"{name} has fewer than {slices_per_volume} slices.")
        kspace_mm[offset : offset + slices_per_volume] = kspace
        maps_mm[offset : offset + slices_per_volume] = espirit_coil_maps(
            kspace, crop=espirit_crop, device=espirit_device
        )
        slice_ids.extend(f"{name}:slice{index}" for index in range(slices_per_volume))
        offset += slices_per_volume
    kspace_mm.flush()
    maps_mm.flush()
    metadata = {
        "version": KSPACE_CACHE_VERSION,
        "data_root": data_root,
        "phase": phase,
        "contrast": contrast,
        "slices_per_volume": int(slices_per_volume),
        "num_slices": int(num_slices),
        "num_volumes": len(names),
        "n_coils": int(n_coils),
        "espirit_center_fraction": ACS_CENTER_FRACTION,
        "espirit_kernel_width": ESPIRIT_KERNEL_WIDTH,
        "espirit_threshold": ESPIRIT_THRESHOLD,
        "espirit_crop": float(espirit_crop),
        "slice_ids": slice_ids,
    }
    with open(paths["meta"], "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def volume_scales(kspace: np.ndarray, coil_maps: np.ndarray, slices_per_volume: int) -> np.ndarray:
    """Per-volume peak ``|A^H y_full|``, broadcast to every slice of the volume."""
    from tqdm import tqdm

    num_slices = int(kspace.shape[0])
    peaks = np.zeros(num_slices, dtype=np.float32)
    for index in tqdm(range(num_slices), desc="volume scales", unit="slice"):
        peaks[index] = float(np.abs(sense_adjoint(kspace[index], coil_maps[index])).max())
    scales = np.ones(num_slices, dtype=np.float32)
    step = max(int(slices_per_volume), 1)
    for start in range(0, num_slices, step):
        peak = float(peaks[start : start + step].max())
        scales[start : start + step] = peak if peak > 0 else 1.0
    return scales


def load_or_build_cache(
    *,
    cache_dir: str,
    data_root: str,
    phase: str,
    contrast: str,
    slices_per_volume: int,
    max_volumes: Optional[int] = None,
    espirit_device=None,
    espirit_crop: float = ESPIRIT_CROP,
    force_rebuild: bool = False,
):
    """Return memory-mapped ``(kspace, coil_maps, scales, slice_ids)``; build on first use."""
    os.makedirs(cache_dir, exist_ok=True)
    digest = _cache_digest(
        data_root=data_root,
        phase=phase,
        contrast=contrast,
        slices_per_volume=int(slices_per_volume),
        max_volumes=max_volumes,
        espirit_crop=float(espirit_crop),
    )
    paths = _cache_paths(cache_dir, digest)
    if force_rebuild:
        for path in paths.values():
            if os.path.isfile(path):
                os.remove(path)
    if not all(os.path.isfile(paths[key]) for key in ("kspace", "coil_maps", "meta")):
        _build_kspace_cache(
            paths,
            data_root=data_root,
            phase=phase,
            contrast=contrast,
            slices_per_volume=int(slices_per_volume),
            max_volumes=max_volumes,
            espirit_device=espirit_device,
            espirit_crop=float(espirit_crop),
        )
    with open(paths["meta"], encoding="utf-8") as handle:
        metadata = json.load(handle)
    kspace = np.load(paths["kspace"], mmap_mode="r")
    coil_maps = np.load(paths["coil_maps"], mmap_mode="r")

    scales = None
    if os.path.isfile(paths["scales"]) and os.path.isfile(paths["scales_meta"]):
        with open(paths["scales_meta"], encoding="utf-8") as handle:
            scales_meta = json.load(handle)
        candidate = np.load(paths["scales"])
        if (
            scales_meta.get("version") == VOLUME_SCALES_CACHE_VERSION
            and candidate.shape[0] == int(metadata["num_slices"])
        ):
            scales = candidate
    if scales is None:
        scales = volume_scales(kspace, coil_maps, int(slices_per_volume))
        np.save(paths["scales"], scales)
        with open(paths["scales_meta"], "w", encoding="utf-8") as handle:
            json.dump(
                {"version": VOLUME_SCALES_CACHE_VERSION, "num_slices": int(metadata["num_slices"])},
                handle,
                indent=2,
            )
    return kspace, coil_maps, scales, list(metadata["slice_ids"])


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def contrast_id_from_slice_id(slice_id: str) -> int:
    """T1/T2/FLAIR label encoded in an M4Raw file name."""
    filename = os.path.basename(str(slice_id).split(":slice", 1)[0])
    try:
        token = filename.split("_", 1)[1].upper()
    except IndexError as exc:
        raise ValueError(f"Cannot parse the M4Raw contrast from {filename!r}.") from exc
    for contrast, contrast_id in CONTRAST_TO_ID.items():
        if token.startswith(contrast):
            return contrast_id
    raise ValueError(f"Unknown M4Raw contrast in {filename!r}; expected T1, T2 or FLAIR.")


class M4RawDataset(Dataset):
    """Paired fully sampled / undersampled M4Raw slices.

    For ``phase="train"`` every slice appears once per training acceleration
    in ``train_r_values``; otherwise each slice is undersampled at ``R``.
    """

    def __init__(
        self,
        *,
        data_root: str,
        cache_dir: str,
        phase: str = "train",
        contrast: str = "all",
        slices_per_volume: int = 18,
        max_volumes: Optional[int] = None,
        train_r_values: Union[str, Sequence[float]] = "4,8",
        R: float = 4.0,
        mask_seed: int = 0,
    ) -> None:
        if phase not in {"train", "val", "test"}:
            raise ValueError("phase must be train, val or test.")
        self.phase = phase
        self.slices_per_volume = int(slices_per_volume)
        self.mask_seed = int(mask_seed)
        self.kspace, self.coil_maps, self.scales, self.slice_ids = load_or_build_cache(
            cache_dir=cache_dir,
            data_root=data_root,
            phase=phase,
            contrast=contrast,
            slices_per_volume=self.slices_per_volume,
            max_volumes=max_volumes,
        )
        self.num_slices = int(self.kspace.shape[0])
        self.height = int(self.kspace.shape[2])
        self.width = int(self.kspace.shape[3])
        if phase == "train":
            self.r_values = tuple(sorted(parse_r_values(train_r_values)))
            self.expand_by_r = True
        else:
            self.r_values = (float(R),)
            self.expand_by_r = False
        self._mask_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.num_slices * (len(self.r_values) if self.expand_by_r else 1)

    def _resolve(self, index: int) -> Tuple[int, float, int]:
        if self.expand_by_r:
            count = len(self.r_values)
            slice_index, r_index = divmod(index, count)
            return slice_index, float(self.r_values[r_index]), self.mask_seed + slice_index * 100 + r_index
        return index, float(self.r_values[0]), self.mask_seed + index

    def sample_id(self, index: int) -> str:
        slice_index, R, _ = self._resolve(int(index))
        return f"{self.slice_ids[slice_index]}:R{R:g}"

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        slice_index, R, seed = self._resolve(int(index))
        if index not in self._mask_cache:
            self._mask_cache[index] = gaussian_mask(self.height, self.width, R, seed=seed)
        mask = self._mask_cache[index]
        kspace = np.array(self.kspace[slice_index], copy=True)
        coil_maps = np.array(self.coil_maps[slice_index], copy=True)
        scale = float(self.scales[slice_index]) or 1.0

        kspace_norm = (kspace / scale).astype(np.complex64, copy=False)
        kspace_us = (kspace_norm * mask[None]).astype(np.complex64, copy=False)
        x_fs = sense_adjoint(kspace_norm, coil_maps)
        x_us = sense_adjoint(kspace_us, coil_maps)
        return {
            "x_fs": torch.from_numpy(complex_to_2ch(x_fs)),
            "x_us": torch.from_numpy(complex_to_2ch(x_us)),
            "y_us": kspace_to_2ch(kspace_us),
            "mask": torch.from_numpy(mask.astype(np.float32)),
            "coil_maps": torch.from_numpy(coil_maps),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "contrast_id": torch.tensor(contrast_id_from_slice_id(self.slice_ids[slice_index]), dtype=torch.long),
            "acceleration": torch.tensor(R, dtype=torch.float32),
            "sample_id": self.sample_id(index),
        }


def collate_mri_batch(samples: List[Dict]) -> Dict:
    """Stack tensors; ``sample_id`` stays a list of strings."""
    batch = {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in samples[0]
        if key != "sample_id"
    }
    batch["sample_id"] = [str(sample["sample_id"]) for sample in samples]
    return batch
