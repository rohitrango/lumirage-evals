from __future__ import annotations

import argparse
import gc
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from fireants.io.image import Image, BatchedImages, FakeBatchedImages
from fireants.registration.moments import MomentsRegistration
from fireants.registration.affine import AffineRegistration

import numpy as np

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

def file_stem_nifti(path: str) -> str:
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return Path(path).stem

def list_processed_subjects(dataroot: Path) -> List[str]:
    mask_paths = sorted(glob(str(dataroot / "brain_mask_processed/*.nii.gz")))
    subjects = [file_stem_nifti(p) for p in mask_paths]
    return subjects

def dice_loss(moved_seg, fixed_seg, eps=1e-5):
    ''' both are tensors '''
    moved_flat = moved_seg.flatten(2)
    fixed_flat = fixed_seg.flatten(2)
    intersection = (moved_flat * fixed_flat).sum(dim=2)
    union = moved_flat.sum(dim=2) + fixed_flat.sum(dim=2)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def ensure_registered_dirs(dataroot: Path) -> None:
    for name in (
        "brain_mask_registered",
        "brain_tissue_registered",
        "brain_subcortical_registered",
        "brain_cerebrum_registered",
    ):
        (dataroot / name).mkdir(parents=True, exist_ok=True)


def subject_paths(dataroot: Path, subject: str) -> Dict[str, str]:
    return {
        "mask": str(dataroot / "brain_mask_processed" / f"{subject}.nii.gz"),
        "tissue": str(dataroot / "brain_tissue_processed" / f"{subject}.nii.gz"),
        "subcortical": str(dataroot / "brain_subcortical_processed" / f"{subject}.nii.gz"),
        "cerebrum": str(dataroot / "brain_cerebrum_processed" / f"{subject}.nii.gz"),
    }


def out_paths(dataroot: Path, subject: str) -> Dict[str, str]:
    return {
        "mask": str(dataroot / "brain_mask_registered" / f"{subject}.nii.gz"),
        "tissue": str(dataroot / "brain_tissue_registered" / f"{subject}.nii.gz"),
        "subcortical": str(dataroot / "brain_subcortical_registered" / f"{subject}.nii.gz"),
        "cerebrum": str(dataroot / "brain_cerebrum_registered" / f"{subject}.nii.gz"),
    }

def get_images_and_masks(
    dataroot: Path,
    fixed_subject: str,
    moving_subject: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return dictionaries of paths for fixed and moving images.

    NOTE: If you prefer to load using fireants IO image wrappers, replace this
    function's implementation accordingly.
    """
    fixed = subject_paths(dataroot, fixed_subject)
    moving = subject_paths(dataroot, moving_subject)
    return fixed, moving


def register_brainmasks(
    fixed_path: str,
    moving_path: str,
):
    """Perform MomentsRegistration then AffineRegistration on cerebrum images.

    Returns a backend-specific transform object. If no backend is available, returns None.

    Backends:
      - fireants (if available and backend in {auto, fireants})
      - SimpleITK (if available and backend in {auto, sitk})
      - None (identity/no-op)
    """
    # Try fireants (placeholder - left for user to implement if desired)
    fixed_batch = BatchedImages([Image.load_file(fixed_path)])
    moving_batch = BatchedImages([Image.load_file(moving_path)])
    # moment_reg = MomentsRegistration(
    #     scale=2,
    #     fixed_images=fixed_batch,
    #     moving_images=moving_batch,
    #     loss_type='fusedmi',
    #     moments=2,
    # )
    # moment_reg.optimize()
    # print(f"Moment reg affine init: {moment_reg.get_affine_init()}")
    affine_reg = AffineRegistration(
        scales=[8, 4, 2, 1],
        iterations=[200, 200, 100, 50],
        fixed_images=fixed_batch,
        moving_images=moving_batch,
        loss_type='custom',
        custom_loss=dice_loss,
        optimizer_lr=3e-3,
        cc_kernel_size=7,
        # init_rigid=moment_reg.get_affine_init(),
    )
    affine_reg.optimize()
    return (affine_reg, fixed_batch)


def save_niftis(affine_reg: AffineRegistration, fixed_batch: BatchedImages, movingp: Dict[str, str], outp: Dict[str, str]) -> None:
    ''' load the images '''
    maskimg = BatchedImages([Image.load_file(movingp["mask"])])
    tissueimg = BatchedImages([Image.load_file(movingp["tissue"], is_segmentation=True, background_seg_label=-1)])
    subcortimg = BatchedImages([Image.load_file(movingp["subcortical"], is_segmentation=True, background_seg_label=-1)])
    cerebrumimg = BatchedImages([Image.load_file(movingp["cerebrum"])])
    print(f"subcortical shape: {subcortimg.shape}, tissue shape: {tissueimg.shape}, cerebrum shape: {cerebrumimg.shape}, mask shape: {maskimg.shape}")

    movedmask = (affine_reg.evaluate(fixed_batch, maskimg).detach() > 0.5).float()
    movedcerebrum = affine_reg.evaluate(fixed_batch, cerebrumimg).detach()
    # need to take argmax
    movedtissue = affine_reg.evaluate(fixed_batch, tissueimg).detach()
    movedsubcort = affine_reg.evaluate(fixed_batch, subcortimg).detach()
    movedtissue = movedtissue.max(dim=1, keepdim=True).indices
    movedsubcort = movedsubcort.max(dim=1, keepdim=True).indices

    FakeBatchedImages(movedmask, fixed_batch).write_image(outp["mask"])
    FakeBatchedImages(movedtissue, fixed_batch).write_image(outp["tissue"])
    FakeBatchedImages(movedsubcort, fixed_batch).write_image(outp["subcortical"])
    FakeBatchedImages(movedcerebrum, fixed_batch).write_image(outp["cerebrum"])
    print(f"Saved moved images to {outp}")


def process(dataroot: Path) -> None:
    ensure_registered_dirs(dataroot)

    subjects = list_processed_subjects(dataroot)
    if len(subjects) == 0:
        raise FileNotFoundError("No processed subjects found under brain_*_processed.")

    fixed_subject = subjects[0]
    print(f"Fixed subject: {fixed_subject}")

    fixed_paths = subject_paths(dataroot, fixed_subject)
    # Save fixed subject images to registered dir
    from shutil import copy2
    fixed_outp = out_paths(dataroot, fixed_subject)
    for key in fixed_paths:
        copy2(fixed_paths[key], fixed_outp[key])

    iterator = tqdm(subjects[1:], desc="Registering subjects", unit="subj")
    for moving_subject in iterator:
        moving_paths = subject_paths(dataroot, moving_subject)

        # Provide paths to get_images_and_masks (user may replace with fireants IO)
        # _fixed, _moving = get_images_and_masks(dataroot, fixed_subject, moving_subject)
        # Currently unused, but kept for API expectation

        # Register cerebrum volumes (fixed vs moving) to compute transform
        affine_reg, fixed_batch = register_brainmasks(
            # fixed_paths["cerebrum"],
            # moving_paths["cerebrum"],
            fixed_paths["mask"],
            moving_paths["mask"],
        )

        # Resample and save all modalities
        outp = out_paths(dataroot, moving_subject)
        save_niftis(affine_reg, fixed_batch, moving_paths, outp)
        # breakpoint()
        del affine_reg, fixed_batch
        gc.collect()

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Affine align processed volumes using moments+affine registration.")
    parser.add_argument("--dataroot", type=Path, required=True, help="Path to dataset root containing brain_*_processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process(args.dataroot)


if __name__ == "__main__":
    main()


