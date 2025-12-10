import os
import glob
import numpy as np
import nibabel as nib
from natsort import natsorted
from pathlib import Path

from vfa.datasets.pairwise_dataset import PairwiseDataset

class OASISDataset(PairwiseDataset):
    def __init__(self, configs, params):
        super().__init__(configs, params)
        self.data_dir = configs['data_dir']
        self.files = natsorted(glob.glob(os.path.join(self.data_dir, "OASIS_OAS*", "aligned_norm.nii.gz")))
        
        # Split into train/val based on configs
        val_size = 20
        if params['func'] == 'train':
            self.files = self.files[:-val_size]
        else:
            self.files = self.files[-val_size:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Get random pair of images
        path = self.files[idx]
        offset = np.random.randint(1, len(self.files))
        mov_idx = (idx + offset) % len(self.files)
        mov_path = self.files[mov_idx]

        # Create sample dict
        sample = {
            'f_img_path': path,
            'f_img': self.load_img_obj(path),
            'm_img_path': mov_path,
            'm_img': self.load_img_obj(mov_path),
        }

        # Add segmentation if available
        f_seg_path = path.replace("aligned_norm.nii.gz", "aligned_seg35.nii.gz")
        m_seg_path = mov_path.replace("aligned_norm.nii.gz", "aligned_seg35.nii.gz")
        
        if os.path.exists(f_seg_path) and os.path.exists(m_seg_path):
            sample.update({
                'f_seg_path': f_seg_path,
                'f_seg': self.load_img_obj(f_seg_path),
                'm_seg_path': m_seg_path,
                'm_seg': self.load_img_obj(m_seg_path)
            })

        # Set prefix for saving results
        if self.params['func'] == 'evaluate':
            f_id = Path(path).parent.name
            m_id = Path(mov_path).parent.name
            sample['prefix'] = os.path.abspath(os.path.join(
                self.params['output_dir'],
                'experiments',
                'oasis',
                f'disp_{f_id}_{m_id}'
            ))

        # Apply transforms
        sample = self.transforms(sample)
        return sample
