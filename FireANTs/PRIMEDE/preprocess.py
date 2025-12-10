from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


def list_brainmask_paths(dataroot: Path) -> List[str]:
    """Return sorted list of brain mask file paths under `brain_mask`.

    The user specified pattern is used exactly: glob(str(dataroot / "brain_mask/*")).
    """
    return sorted(glob(str(dataroot / "brain_mask/*nii.gz")))

def derive_related_paths(brainmask_paths: Sequence[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Derive tissue, subcortical, cerebrum paths from brainmask paths.

    Returns tuples of lists in the order: (brainmask, tissue, subcortical, cerebrum)
    """
    brainmask = list(brainmask_paths)
    tissue = [p.replace("brain_mask", "brain_tissue") for p in brainmask]
    subcortical = [p.replace("brain_mask", "brain_subcortical") for p in brainmask]
    cerebrum = [p.replace("brain_mask", "brain_cerebrum") for p in brainmask]
    return brainmask, tissue, subcortical, cerebrum

def is_nifti(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")

class ImageMeta:
    """Container for image metadata for saving back out.

    For NIfTI files, stores `affine` and `header`.
    """

    def __init__(self, affine: Optional[np.ndarray] = None, header: Optional[object] = None):
        self.affine = affine
        self.header = header

import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
from nibabel.orientations import axcodes2ornt, ornt_transform

def reorient_img(img: nib.Nifti1Image, ori="RAS") -> nib.Nifti1Image:
     ori = axcodes2ornt(ori)
     cur_ori = axcodes2ornt(nib.aff2axcodes(img.affine))
     trans_ornt = ornt_transform(cur_ori, ori)
     img = img.as_reoriented(trans_ornt)
     return img

def load_image(path: str) -> Tuple[np.ndarray, ImageMeta]:
    """Load image as numpy array along with metadata needed for saving.

    Supports .nii/.nii.gz via nibabel (if installed).
    """
    if is_nifti(path):
        try:
            import nibabel as nib  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Reading NIfTI requires nibabel. Please install nibabel."
            ) from exc
        img = nib.load(path)
        img = reorient_img(img, ori="RAS")

        data = img.get_fdata(dtype=np.float32)
        # Preserve original dtype if integer-like masks
        if np.issubdtype(img.get_data_dtype(), np.integer):
            data = img.get_fdata(dtype=np.float32)  # keep float32 but note below we can cast on save
        return data, ImageMeta(img.affine, img.header)

    print(f"Error: only NIfTI (.nii/.nii.gz) files are supported: {path}")
    raise ValueError(f"Unsupported file type (expected NIfTI): {path}")


def save_image(array: np.ndarray, meta: ImageMeta, out_path: str, like_path: Optional[str] = None) -> None:
    """Save array to out_path.

    - If out_path ends with .nii/.nii.gz, saves via nibabel with spacing of 1 and origin of 0
      (identity affine). Original metadata is intentionally ignored for outputs.
    """
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if is_nifti(out_path):
        try:
            import nibabel as nib  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Writing NIfTI requires nibabel. Please install nibabel."
            ) from exc

        # Use identity affine: spacing = 1, origin = 0
        affine = np.eye(4, dtype=float)

        # Choose output dtype: preserve integer types for masks/labels if array is integral
        output_dtype = np.float32
        if np.all(np.mod(array, 1) == 0):  # simple heuristic for label-like data
            output_dtype = np.int16
        img = nib.Nifti1Image(array.astype(output_dtype), affine)

        # Ensure header zooms are 1 and forms are identity
        hdr = img.header
        # Set voxel sizes to 1 for spatial dimensions
        try:
            hdr.set_zooms((1.0, 1.0, 1.0))
        except Exception:
            pass
        try:
            img.set_qform(affine, code=1)
            img.set_sform(affine, code=1)
        except Exception:
            pass
        nib.save(img, out_path)
        return

    print(f"Error: only NIfTI (.nii/.nii.gz) outputs are supported: {out_path}")
    raise ValueError(f"Unsupported output file type (expected NIfTI): {out_path}")


def compute_bbox(mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compute bounding box (mins, sizes) for non-zero voxels in a 3D mask.

    Returns None if mask is empty (all zeros).
    mins and sizes are 1D arrays of length ndim.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask.shape}")
    indices = np.argwhere(mask > 0)
    if indices.size == 0:
        return None
    mins = indices.min(axis=0)
    maxs = indices.max(axis=0)
    sizes = (maxs - mins) + 1
    return mins, sizes


def round_up_to_multiple(values: np.ndarray, multiple: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    return ((values + multiple - 1) // multiple) * multiple


def crop_to_bbox(array: np.ndarray, mins: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    slices = tuple(slice(int(m), int(m + s)) for m, s in zip(mins, sizes))
    return array[slices]


def pad_to_shape_center(array: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Zero-pad a 3D array to target_shape, centering the content.
    """
    target_shape = tuple(int(x) for x in target_shape)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array for padding, got shape {array.shape}")
    pad_width = []
    for curr, target in zip(array.shape, target_shape):
        if curr > target:
            raise ValueError(f"Cannot pad: current dim {curr} exceeds target {target}")
        total = target - curr
        before = total // 2
        after = total - before
        pad_width.append((before, after))
    padded = np.pad(array, pad_width=pad_width, mode="constant", constant_values=0)
    return padded


def compute_max_bbox_size(brainmask_paths: Sequence[str]) -> np.ndarray:
    """Scan all masks and compute the maximum bbox size across images (unrounded)."""
    max_sizes = None
    for p in tqdm(brainmask_paths, desc="Scanning masks", unit="img"):
        mask, _ = load_image(p)
        bbox = compute_bbox(mask)
        if bbox is None:
            # Skip empty mask; log and continue
            print(f"Warning: empty mask (all zeros), skipping for max-size computation: {p}")
            continue
        _, sizes = bbox
        if max_sizes is None:
            max_sizes = sizes.astype(np.int64)
        else:
            max_sizes = np.maximum(max_sizes, sizes)
    if max_sizes is None:
        raise RuntimeError("No valid masks with non-zero voxels were found.")
    return max_sizes


def ensure_processed_dirs(dataroot: Path) -> None:
    for name in ("brain_mask_processed", "brain_tissue_processed", "brain_subcortical_processed", "brain_cerebrum_processed"):
        (dataroot / name).mkdir(parents=True, exist_ok=True)


def replace_dir(path: str, src_dir_name: str, dst_dir_name: str, dataroot: Path) -> str:
    """Replace one leaf directory name with another under the same dataroot."""
    rel = str(Path(path).relative_to(dataroot))
    new_rel = rel.replace(src_dir_name, dst_dir_name, 1)
    return str(dataroot / new_rel)


def process_dataset(
    dataroot: Path,
    multiple_of: int = 32,
) -> None:
    # 1) Collect brain masks
    brainmask_paths = list_brainmask_paths(dataroot)
    if len(brainmask_paths) == 0:
        raise FileNotFoundError(f"No brain masks found under: {dataroot / 'brain_mask'}")

    # 2) Compute maximum bbox size across all masks
    print("Scanning masks to compute maximum bounding-box size...")
    max_sizes = compute_max_bbox_size(brainmask_paths)
    target_shape = tuple(int(x) for x in round_up_to_multiple(max_sizes, multiple_of))
    print(f"Max bbox size (unrounded): {tuple(int(x) for x in max_sizes)}")
    print(f"Target shape (rounded to {multiple_of}): {target_shape}")

    # 3) Build related file lists
    brainmask, tissue, subcortical, cerebrum = derive_related_paths(brainmask_paths)

    # 4) Prepare output directories
    ensure_processed_dirs(dataroot)

    # 5) Process each tuple
    print("Processing images: crop to per-image bbox then pad to target shape...")
    iterator = zip(brainmask, tissue, subcortical, cerebrum)
    iterator = tqdm(iterator, total=len(brainmask), desc="Processing images", unit="img")
    for i, (m_path, t_path, s_path, c_path) in enumerate(iterator, start=1):
        try:
            mask_arr, mask_meta = load_image(m_path)
        except Exception as exc:
            print(f"Error reading mask '{m_path}': {exc}")
            continue

        bbox = compute_bbox(mask_arr)
        if bbox is None:
            print(f"Warning: empty mask, skipping sample: {m_path}")
            continue
        mins, sizes = bbox

        # Load other related images
        try:
            tissue_arr, tissue_meta = load_image(t_path)
        except Exception as exc:
            print(f"Error reading tissue '{t_path}': {exc}")
            continue
        try:
            subcort_arr, subcort_meta = load_image(s_path)
        except Exception as exc:
            print(f"Error reading subcortical '{s_path}': {exc}")
            continue
        try:
            cerebrum_arr, cerebrum_meta = load_image(c_path)
        except Exception as exc:
            print(f"Error reading cerebrum '{c_path}': {exc}")
            continue

        # Sanity check shapes
        for name, arr in (
            ("mask", mask_arr),
            ("tissue", tissue_arr),
            ("subcortical", subcort_arr),
            ("cerebrum", cerebrum_arr),
        ):
            if arr.shape != mask_arr.shape:
                print(
                    f"Warning: shape mismatch for {name}. got {arr.shape}, expected {mask_arr.shape} based on mask."
                )

        # Crop to bbox
        mask_crop = crop_to_bbox(mask_arr, mins, sizes)
        tissue_crop = crop_to_bbox(tissue_arr, mins, sizes)
        subcort_crop = crop_to_bbox(subcort_arr, mins, sizes)
        cerebrum_crop = crop_to_bbox(cerebrum_arr, mins, sizes)

        # Pad to target shape
        mask_proc = pad_to_shape_center(mask_crop, target_shape)
        tissue_proc = pad_to_shape_center(tissue_crop, target_shape)
        subcort_proc = pad_to_shape_center(subcort_crop, target_shape)
        cerebrum_proc = pad_to_shape_center(cerebrum_crop, target_shape)

        # Determine output paths
        out_mask = replace_dir(m_path, "brain_mask", "brain_mask_processed", dataroot)
        out_tissue = replace_dir(t_path, "brain_tissue", "brain_tissue_processed", dataroot)
        out_subcort = replace_dir(s_path, "brain_subcortical", "brain_subcortical_processed", dataroot)
        out_cerebrum = replace_dir(c_path, "brain_cerebrum", "brain_cerebrum_processed", dataroot)

        # Save
        try:
            save_image(mask_proc, mask_meta, out_mask, like_path=m_path)
            save_image(tissue_proc, tissue_meta, out_tissue, like_path=t_path)
            save_image(subcort_proc, subcort_meta, out_subcort, like_path=s_path)
            save_image(cerebrum_proc, cerebrum_meta, out_cerebrum, like_path=c_path)
        except Exception as exc:
            print(f"Error saving processed outputs for '{m_path}': {exc}")
            continue

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess brain volumes: crop to mask bbox and pad to max size.")
    parser.add_argument("--dataroot", type=Path, help="Path to dataset root containing brain_* directories")
    parser.add_argument(
        "--multiple-of",
        type=int,
        default=32,
        help="Round the maximum bbox size up to this multiple (default: 32)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(args.dataroot, multiple_of=args.multiple_of)


if __name__ == "__main__":
    main()


