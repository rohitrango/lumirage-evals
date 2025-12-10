from dataset import UltracortexInterSubjectDataset, UltracortexDataset, ultracortex_collate_fn
from fireants.registration.moments import MomentsRegistration
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.syn import SyNRegistration
from mind import MINDSSC
import argparse

import torch
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

def dice_scores(moved_seg, fixed_seg, labels, eps=1e-5):
    ''' both are tensors '''
    dices = []
    for lab in labels:
        movedl = moved_seg == lab
        fixedl = fixed_seg == lab
        intersection = (movedl * fixedl).sum()
        union = movedl.sum() + fixedl.sum()
        dice = (2 * intersection + eps) / (union + eps)
        dices.append(dice)
    return torch.stack(dices)

def main(args):
    dataset = UltracortexDataset(
        json_path=args.json_path,
        nearest=True
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=ultracortex_collate_fn)
    print(f"Number of batches: {len(dataloader)}")
    if len(dataloader) == 0:
        raise ValueError("No batches found")

    all_dice = []
    avg_dice = 0.0
    avg_count = 0

    for batch in tqdm(dataloader):
        if args.num_samples is not None and avg_count >= args.num_samples:
            break
        fixed_img = batch['fixed_img']
        moving_img = batch['moving_img']
        fixed_seg = batch['fixed_seg']
        moving_seg = batch['moving_seg']

        fmod = batch['f_modality'][0]
        mmod = batch['m_modality'][0]
        loss_type = "fusedcc"

        # apply MIND descriptor if multimodal
        if args.baseline:
            deform = 1
            moved_seg = moving_seg()
        else:
            if fmod == mmod:
                pass
            else:
                fixed_img.batch_tensor = MINDSSC(fixed_img.batch_tensor)
                moving_img.batch_tensor = MINDSSC(moving_img.batch_tensor)

            print(f"Fixed image shape: {fixed_img.batch_tensor.shape}, Moving image shape: {moving_img.batch_tensor.shape}")
            print(f"Registering {batch['fixed_img_path']} and {batch['moving_img_path']}")
            print(f"Fixed seg: {batch['fixed_seg_path']}, Moving seg: {batch['moving_seg_path']}")
            print(f"fixed image shape: {fixed_img.shape}, moving image shape: {moving_img.shape}")
            print(f"Fixed modality: {fmod}, moving modality: {mmod}, loss_type: {loss_type}")

            moments_reg = MomentsRegistration(
                scale=2,
                fixed_images=fixed_img,
                moving_images=moving_img,
                loss_type='fusedmi',
                moments=1,
            )
            moments_reg.optimize()

            # greedy
            DeformableRegistration = GreedyRegistration if args.greedy else SyNRegistration
            deform = DeformableRegistration(
                scales=args.deformable_scales,
                iterations=args.deformable_iterations,
                fixed_images=fixed_img,
                moving_images=moving_img,
                loss_type=loss_type,
                loss_params={'smooth_nr': 1e-5, 'smooth_dr': 1e-5} if loss_type == "fusedcc" else {},
                cc_kernel_size=7,
                optimizer_lr=args.deformable_lr,
                max_tolerance_iters=10,
                init_affine=moments_reg.get_affine_init(),
            )
            deform.optimize()

            moved_seg = deform.evaluate(fixed_img, moving_seg).detach()
        # Get labels from dataset
        labels = dataset.labels
        labelrange = np.arange(len(labels)) + 1
        dice = dice_scores(moved_seg.max(dim=1, keepdim=True).indices, fixed_seg().max(dim=1, keepdim=True).indices, labelrange)
        init_dice = dice_scores(fixed_seg().max(dim=1, keepdim=True).indices, moving_seg().max(dim=1, keepdim=True).indices, labelrange)
        # add to avg dice
        avg_dice += dice.mean().item()
        avg_count += 1
        print(f"\n\nInit dice score: {init_dice.mean().item()}, shape: {init_dice.shape}, init_dice: {init_dice}")
        print(f"Dice score: {dice.mean().item()}, shape: {dice.shape}, dice: {dice}")
        print(f"Average dice: {avg_dice / avg_count}\n\n")
        all_dice.append(dice.cpu().numpy())

        del deform
        del fixed_img, moving_img, fixed_seg, moving_seg, moved_seg
        torch.cuda.empty_cache()
        gc.collect()


    all_dice = np.stack(all_dice, axis=0)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.baseline:
        np.save(os.path.join(args.output_dir, f"baseline_dice_scores.npy"), all_dice)
    else:
        np.save(os.path.join(args.output_dir, f"{'greedy' if args.greedy else 'syn'}_dice_scores.npy"), all_dice)
    print("Average Dice score: ", all_dice.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON file with dataset pairs")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--affine_scales", type=str, default="4,2,1")
    parser.add_argument("--affine_iterations", type=str, default="200, 100, 30")
    parser.add_argument("--deformable_scales", type=str, default="12, 8, 4, 2, 1")
    parser.add_argument("--deformable_iterations", type=str, default="200, 200, 100, 100, 20")
    parser.add_argument("--deformable_lr", type=float, default=0.75)
    parser.add_argument("--no-greedy", dest="greedy", action="store_false")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--baseline", action='store_true')
    args = parser.parse_args()
    args.affine_scales = list(map(int, args.affine_scales.split(",")))
    args.affine_iterations = list(map(int, args.affine_iterations.split(",")))
    args.deformable_scales = list(map(int, args.deformable_scales.split(",")))
    args.deformable_iterations = list(map(int, args.deformable_iterations.split(",")))
    if not args.greedy:
        args.deformable_lr *= 0.5

    main(args)
