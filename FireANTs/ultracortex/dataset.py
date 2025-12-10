from webbrowser import BackgroundBrowser
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from fireants.io.image import Image, BatchedImages
from glob import glob
from natsort import natsorted
import os
import json
from functools import partial
from typing import List

def seg_contiguous_preprocessor(array: torch.Tensor, labels: List[int]) -> torch.Tensor:
    ''' get contiguous segmentation mask '''
    if 0 not in labels:
        labels = [0] + labels
    new_array = torch.zeros_like(array)
    for newlabel, label in enumerate(labels):
        new_array[array == label] = newlabel
    return new_array

def ultracortex_collate_fn(batch):
    # Collate images into BatchedImages, everything else default
    batched = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], Image):
            # Convert list of Images to BatchedImages
            batched[key] = BatchedImages([item[key] for item in batch])
        else:
            # Default collation for non-Image data
            batched[key] = torch.utils.data.dataloader.default_collate([item[key] for item in batch])
    return batched

ignore_subjects = [37, 45, 57]

class UltracortexInterSubjectDataset(Dataset):
    def __init__(self, data_dir: str, seg_dir: str = "manual_segmentation", img_dir: str = "skullstrips", background_label: int = 0, nearest: bool = False):
        self.data_dir = data_dir
        self.segmentations = natsorted(glob(os.path.join(data_dir, seg_dir, '*.nii')))
        # filter out subjects in ignore_subjects
        def filter_func(path):
            if any([f"sub-{s}" in path for s in ignore_subjects]):
                return False
            return True
        self.segmentations = list(filter(filter_func, self.segmentations))

        self.images = [x.replace(seg_dir, img_dir).replace("_seg", "_skullstrip") for x in self.segmentations]
        assert all(os.path.exists(x) for x in self.images), "Brain masks not found"
        self.n = len(self.segmentations)
        self.background_label = background_label
        self.nearest = nearest

    def __len__(self):
        return self.n * (self.n - 1)

    def __getitem__(self, idx):
        i, j = idx // self.n, idx % (self.n - 1) + 1   # i is from 0 to n-1, j is from 1 to n-1 (offset)
        j = (i + j) % self.n
        fixedimg = Image.load_file(self.images[i])
        movingimg = Image.load_file(self.images[j])
        if self.nearest:
            # use integer labels for nearest neighbor interpolation
            fixedseg = Image.load_file(self.segmentations[i], is_segmentation=True, seg_preprocessor=seg_contiguous_preprocessor, background_seg_label=-1)
            movingseg = Image.load_file(self.segmentations[j], is_segmentation=True, seg_preprocessor=seg_contiguous_preprocessor, background_seg_label=-1)
        else:
            fixedseg = Image.load_file(self.segmentations[i], is_segmentation=True, seg_preprocessor=seg_contiguous_preprocessor, background_seg_label=self.background_label)
            movingseg = Image.load_file(self.segmentations[j], is_segmentation=True, seg_preprocessor=seg_contiguous_preprocessor, background_seg_label=self.background_label)
        return {
            'pair_idx': (i, j),
            'fixed_img_path': self.images[i],
            'moving_img_path': self.images[j],
            'fixed_seg_path': self.segmentations[i],
            'moving_seg_path': self.segmentations[j],
            'fixed_img': fixedimg,
            'moving_img': movingimg,
            'fixed_seg': fixedseg,
            'moving_seg': movingseg,
        }

    def get_labels(self, pair_idx):
        """Get unique labels for a given pair index"""
        i, j = pair_idx // self.n, pair_idx % (self.n - 1) + 1
        j = (i + j) % self.n
        # Load both segmentations and get unique labels
        fixed_seg = sitk.ReadImage(self.segmentations[i])
        moving_seg = sitk.ReadImage(self.segmentations[j])
        fixed_array = sitk.GetArrayFromImage(fixed_seg)
        moving_array = sitk.GetArrayFromImage(moving_seg)
        import numpy as np
        labels = np.unique(np.concatenate([fixed_array.flatten(), moving_array.flatten()]))
        # Apply seg_contiguous_preprocessor logic to get contiguous labels
        return torch.arange(len(labels)).tolist()


class UltracortexDataset(Dataset):
    """Torch dataset for Ultracortex data from JSON files."""
    def __init__(
        self,
        json_path: str,
        nearest: bool = False,
        background_label: int = 0,
    ) -> None:
        super().__init__()
        self.json_path = json_path
        self.nearest = nearest
        self.background_label = background_label
        with open(self.json_path, "r") as f:
            data = json.load(f)
            self.data = data['pairs']
            self.labels = data['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        fixedimg = Image.load_file(datum['f_img'])
        movingimg = Image.load_file(datum['m_img'])
        if self.nearest:
            # use integer labels for nearest neighbor interpolation
            fixedseg = Image.load_file(datum['f_seg'], is_segmentation=True, seg_preprocessor=partial(seg_contiguous_preprocessor, labels=self.labels), background_seg_label=-1)
            movingseg = Image.load_file(datum['m_seg'], is_segmentation=True, seg_preprocessor=partial(seg_contiguous_preprocessor, labels=self.labels), background_seg_label=-1)
        else:
            fixedseg = Image.load_file(datum['f_seg'], is_segmentation=True, seg_preprocessor=partial(seg_contiguous_preprocessor, labels=self.labels), background_seg_label=self.background_label)
            movingseg = Image.load_file(datum['m_seg'], is_segmentation=True, seg_preprocessor=partial(seg_contiguous_preprocessor, labels=self.labels), background_seg_label=self.background_label)
        return {
            'pair_idx': idx,
            'fixed_img_path': datum['f_img'],
            'moving_img_path': datum['m_img'],
            'fixed_seg_path': datum['f_seg'],
            'moving_seg_path': datum['m_seg'],
            'f_modality': datum['f_modality'],
            'm_modality': datum['m_modality'],
            'fixed_img': fixedimg,
            'moving_img': movingimg,
            'fixed_seg': fixedseg,
            'moving_seg': movingseg,
        }
