import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from functools import partial

from fireants.io.image import Image, BatchedImages
import json

def primede_collate_fn(batch: List[Dict]):
    batched = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], Image):
            batched[key] = BatchedImages([item[key] for item in batch])
        else:
            batched[key] = torch.utils.data.dataloader.default_collate([item[key] for item in batch])
    return batched

class PRIMEDEDataset(Dataset):
    """Torch dataset for NIMH data tree.
    """
    def __init__(
        self,
        json_path: str,
        nearest: bool = False,
    ) -> None:
        super().__init__()
        self.json_path = json_path
        self.nearest = nearest
        with open(self.json_path, "r") as f:
            data = json.load(f)
            self.data = data['pairs']
            self.labels = data['labels']
            print(f"Loaded {len(self.data)} pairs with labels {self.labels}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        datum = self.data[idx]
        fixedimg = Image.load_file(datum['f_img'])
        movingimg = Image.load_file(datum['m_img'])
        if self.nearest:
            # use integer labels
            fixedseg = Image.load_file(datum['f_seg'], is_segmentation=True, max_seg_label=self.labels[-1], background_seg_label=-1)
            movingseg = Image.load_file(datum['m_seg'], is_segmentation=True, max_seg_label=self.labels[-1], background_seg_label=-1)
        else:
            fixedseg = Image.load_file(datum['f_seg'], is_segmentation=True, max_seg_label=self.labels[-1])
            movingseg = Image.load_file(datum['m_seg'], is_segmentation=True, max_seg_label=self.labels[-1])

        return {
            'pair_idx': idx,
            'fixed_img_path': datum['f_img'],
            'moving_img_path': datum['m_img'],
            'fixed_seg_path': datum['f_seg'],
            'moving_seg_path': datum['m_seg'],
            'fixed_img': fixedimg,
            'moving_img': movingimg,
            'fixed_seg': fixedseg,
            'moving_seg': movingseg,
        }
