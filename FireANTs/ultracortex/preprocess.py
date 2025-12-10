from fireants.io.image import FakeBatchedImages
from torch._dynamo.external_utils import FakeBackwardCFunction
from dataset import UltracortexInterSubjectDataset, ultracortex_collate_fn
from fireants.registration.moments import MomentsRegistration
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

def get_int_label(seg):
    ''' BCHWD -> BHWD '''
    return seg.max(dim=1, keepdim=True).indices

def winsorize(img):
    imgnp = img.cpu().numpy()
    m = np.percentile(imgnp, 1)
    M = np.percentile(imgnp, 99)
    img = torch.clamp(img, m, M)
    img = (img - m) / (M - m)
    return img

def main():
    data_dir = "/mnt/rohit_data2/ultracortex/derivatives"
    imgsavedir = os.path.join(data_dir, "processed_skullstrips")
    segsavedir = os.path.join(data_dir, "processed_segmentations")
    os.makedirs(imgsavedir, exist_ok=True)
    os.makedirs(segsavedir, exist_ok=True)

    # the idea is to bring all images to the same voxel size and the data to be in the same space
    dataset = UltracortexInterSubjectDataset(data_dir=data_dir, background_label=-1)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=ultracortex_collate_fn, shuffle=False)
    n = dataset.n
    print(f"Number of batches: {len(dataloader)}")
    if len(dataloader) == 0:
        raise ValueError("No batches found")
    
    idx = 0
    pbar = tqdm(range(n))

    for batch in dataloader:
        pbar.update(1)
        if idx >= n:
            break
        idx += 1    
        fixed_img = batch['fixed_img']
        moving_img = batch['moving_img']
        fixed_seg = batch['fixed_seg']
        moving_seg = batch['moving_seg']

        breakpoint()

        # save fixed image
        if idx == 1:
            fixed_img_name = batch['fixed_img_path'][0].split("/")[-1]
            FakeBatchedImages(winsorize(fixed_img()), fixed_img).write_image(os.path.join(imgsavedir, fixed_img_name))
            # save seg
            fixed_seg_name = batch['fixed_seg_path'][0].split("/")[-1]
            FakeBatchedImages(get_int_label(fixed_seg()), fixed_img).write_image(os.path.join(segsavedir, fixed_seg_name))

            print(f"Saved fixed image to {os.path.join(imgsavedir, fixed_img_name)}")

        print(f"Fixed seg: {batch['fixed_seg_path']}, Moving seg: {batch['moving_seg_path']}")

        moment_reg = MomentsRegistration(
            scale=2,
            fixed_images=fixed_seg,
            moving_images=moving_seg,
            loss_type='fusedcc',
            moments=1,
        )
        moment_reg.optimize()

        # we need to save the new images
        # moved_seg = moment_reg.evaluate(fixed_img, moving_seg).detach()
        # moved_seg = moved_seg.max(dim=1).indices
        moved_img_name = batch['moving_img_path'][0].split("/")[-1]
        moved_img = moment_reg.evaluate(fixed_img, moving_img).detach()
        FakeBatchedImages(winsorize(moved_img), fixed_img).write_image(os.path.join(imgsavedir, moved_img_name))

        moved_seg_name = batch['moving_seg_path'][0].split("/")[-1]
        moved_seg = moment_reg.evaluate(fixed_seg, moving_seg).detach()
        FakeBatchedImages(get_int_label(moved_seg), fixed_img).write_image(os.path.join(segsavedir, moved_seg_name))
        print(f"Saved moved image to {os.path.join(imgsavedir, moved_img_name)}")
        print(f"Saved moved seg to {os.path.join(segsavedir, moved_seg_name)}")


if __name__ == "__main__":
    main()
