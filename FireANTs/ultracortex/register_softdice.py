from dataset import UltracortexInterSubjectDataset, UltracortexDataset, ultracortex_collate_fn
from fireants.registration.moments import MomentsRegistration
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration
import argparse

import torch
from mind import MINDSSC
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

def dice_scores(moved_seg, fixed_seg, eps=1e-5):
    ''' both are tensors '''
    moved_flat = moved_seg.flatten(2)
    fixed_flat = fixed_seg.flatten(2)
    intersection = (moved_flat * fixed_flat).sum(dim=2)
    union = moved_flat.sum(dim=2) + fixed_flat.sum(dim=2)
    dice = (2 * intersection + eps) / (union + eps)
    return dice

def main(args):
    dataset = UltracortexDataset(json_path=args.json_path)

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=ultracortex_collate_fn)
    print(f"Number of batches: {len(dataloader)}")
    if len(dataloader) == 0:
        raise ValueError("No batches found")

    all_dice = []
    avg_dice = 0.0
    avg_count = 0

    for batch in tqdm(dataloader):
        fixed_img = batch['fixed_img']
        moving_img = batch['moving_img']
        fixed_seg = batch['fixed_seg']
        moving_seg = batch['moving_seg']

        fmod = batch['f_modality'][0]
        mmod = batch['m_modality'][0]
        loss_type = "fusedcc" 

        if fmod == mmod:
            pass
        else:
            fixed_img.batch_tensor = MINDSSC(fixed_img.batch_tensor)
            moving_img.batch_tensor = MINDSSC(moving_img.batch_tensor)

        print(f"Registering {batch['fixed_img_path']} and {batch['moving_img_path']}")
        print(f"Fixed seg: {batch['fixed_seg_path']}, Moving seg: {batch['moving_seg_path']}")
        print(f"fixed image shape: {fixed_img.shape}, moving image shape: {moving_img.shape}")
        print(f"Fixed modality: {fmod}, moving modality: {mmod}, loss_type: {loss_type}")

        affine_reg = AffineRegistration(
            scales=args.affine_scales,
            iterations=args.affine_iterations,
            fixed_images=fixed_img,
            moving_images=moving_img,
            loss_type=loss_type,
            loss_params={'smooth_nr': 1e-5, 'smooth_dr': 1e-5} if loss_type == "fusedcc" else {},
            cc_kernel_size=7,
            tolerance=1e-4,
        )
        affine_reg.optimize()

        # greedy
        DeformableRegistration = GreedyRegistration if args.greedy else SyNRegistration
        deform = DeformableRegistration(
            scales=args.deformable_scales,
            iterations=args.deformable_iterations,
            fixed_images=fixed_img,
            moving_images=moving_img,
            init_affine=affine_reg.get_affine_matrix(),
            loss_type=loss_type,
            loss_params={'smooth_nr': 1e-5, 'smooth_dr': 1e-5} if loss_type == "fusedcc" else {},
            cc_kernel_size=7,
            optimizer_lr=args.deformable_lr,
            # smooth_warp_sigma=0.75,
            # smooth_grad_sigma=1.5,
        )
        deform.optimize()

        # evaluate
        moved_seg = deform.evaluate(fixed_img, moving_seg).detach()
        dice = dice_scores(moved_seg, fixed_seg())
        init_dice = dice_scores(fixed_seg(), moving_seg())
        # add to avg dice
        avg_dice += dice.mean().item()
        avg_count += 1
        print(f"\n\nInit dice score: {init_dice.mean().item()}, shape: {init_dice.shape}, init_dice: {init_dice}")
        print(f"Average dice: {avg_dice / avg_count}")
        print(f"Dice score: {dice.mean().item()}, shape: {dice.shape}, dice: {dice}\n\n")
        all_dice.append(dice.cpu().numpy())

        del affine_reg, deform
        del fixed_img, moving_img, fixed_seg, moving_seg, moved_seg
        torch.cuda.empty_cache()
        gc.collect()

    
    all_dice = np.concatenate(all_dice, axis=0)
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, f"{'greedy' if args.greedy else 'syn'}_dice_scores.npy"), all_dice)
    print("Average Dice score: ", all_dice.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON file with dataset pairs")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--affine_scales", type=list, default=[4, 2, 1])
    parser.add_argument("--affine_iterations", type=list, default=[200, 100, 30])
    parser.add_argument("--deformable_scales", type=list, default=[6, 4, 2, 1])
    parser.add_argument("--deformable_iterations", type=list, default=[200, 200, 100, 100])
    parser.add_argument("--deformable_lr", type=float, default=0.75)
    parser.add_argument("--no-greedy", dest="greedy", action="store_false")
    args = parser.parse_args()
    main(args)
