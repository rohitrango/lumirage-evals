import torch.nn as nn
import torch
import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
import time
import socket
import os
import numpy as np
import pathlib
import argparse
import json
import pdb
import logging
from datetime import datetime
from tqdm import tqdm
from contextlib import nullcontext
import wandb
import gc
from vfa.models import create_network_class
from vfa.datasets import load_dataset
from vfa.utils.utils import update_params_json, update_params_args, setup_logging, load_data_configs
from vfa.timer import Timer
setup_logging()
logger = logging.getLogger(__name__)

@torch.inference_mode(False)
def train(net, trainDL, params, optimizer, scheduler, scaler, pbar, data_configs):
    net.train()
    epoch_loss = {x:[] for x in params['loss']['train']}

    for sample in trainDL:
        sample_loss = {x:0 for x in params['loss']['train']}
        # initialize timer for each sample
        #timer = Timer(sync=params.get('sync', True))
        timer = Timer(sync=True)

        with autocast(dtype=torch.float16) if params['fp16'] else nullcontext():
            with timer.get_context("forward_pass"):
                results = net(sample)
                total_loss = 0
            
            for loss in params['loss']['train']:
                with timer.get_context(f"loss_computation_{loss.lower()}"):
                    loss_function_name = f"calc_{loss.lower()}_loss"
                    loss_function = getattr(net, loss_function_name, None)
                    if loss_function is not None:
                        sample_loss[loss] = loss_function(
                                                results=results,
                                                sample=sample,
                                                phase='train',
                                                labels=data_configs['labels']
                        )
                        sample_loss[loss] *= params['loss']['train'][loss]['weight']
                    else:
                        raise NotImplementedError(f'{loss_function} not implemented')
                    pass

                with timer.get_context(f"loss_aggregation"):
                    epoch_loss[loss].append(sample_loss[loss].item())
                    if loss in params['loss']['train']:
                        total_loss += sample_loss[loss]
                    pass

        with timer.get_context("backward_pass"):
            optimizer.zero_grad()
            if params['fp16']:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            scheduler.step()
        
        timer.get_stats()

        pbar.update(1)

    return epoch_loss

@torch.inference_mode()
def evaluate(net, evalDL, params, pbar, data_configs, num_samples=None):
    net.eval()
    epoch_loss = {x:[] for x in params['loss']['eval']}
    sample_id = 0

    try:
        for sample in evalDL:
            if num_samples is not None and sample_id >= num_samples:
                break
            sample_id += 1
            timer = Timer(sync=params.get('sync', True))
            sample_loss = {x:0 for x in params['loss']['eval']}

            if params.get('resolution') is None:
                with timer.get_context("eval_forward_pass"):
                    results = net(sample)

                with timer.get_context("eval_loss_computation"):
                    # calculate losses
                    for loss in params['loss']['eval']:
                        loss_function_name = f"calc_{loss.lower()}_loss"
                        loss_function = getattr(net, loss_function_name, None)
                        if loss_function is not None:
                            sample_loss[loss] = loss_function(
                                                    results=results,
                                                    sample=sample,
                                                    phase='eval',
                                                    labels=data_configs['labels']
                            )
                            if loss.lower() == 'dice':
                                print(f"dice: {sample_loss[loss].cpu().numpy()}")
                            print(f"loss: {loss} --- {sample_loss[loss].cpu().mean().item()}")
                            sample_loss[loss] *= params['loss']['eval'][loss]['weight']
                        else:
                            raise NotImplementedError(f'{loss_function_name} not implemented')

                        epoch_loss[loss].append(sample_loss[loss].cpu().numpy())
            else:
                # ran forward patches
                with timer.get_context("eval_forward_patches"):
                    results = net.forward_patches(sample)
            
            timer.get_stats()

            # print average dice score
            print(f"average dice: {np.mean(np.stack(epoch_loss['Dice']))}")

            evalDL.dataset.export(results, sample)
            pbar.update(1)

    except KeyboardInterrupt:
        pass

    return epoch_loss

#### Batch size eval
# @torch.inference_mode()
# def evaluate(net, evalDL, params, pbar, data_configs, num_samples=None):
#     net.eval()
#     epoch_loss = {x:[] for x in params['loss']['eval']}
#     sample_id = 0

#     batch_size = 1
#     runtimes = []

#     try:
#         for sample in evalDL:
#             break
#         torch.cuda.empty_cache()

#         while True:
#             timer = Timer(sync=params.get('sync', True))
#             sample_loss = {x:0 for x in params['loss']['eval']}
#             batch_sample = {}
#             for k, v in sample.items():
#                 if isinstance(v, torch.Tensor):
#                     repeat = [batch_size] + [1 for _ in range(v.dim() - 1)]
#                     batch_sample[k] = v.clone().repeat(repeat)
#                 else:
#                     batch_sample[k] = v

#             a = time.perf_counter()
#             print(f"batch_sample: {batch_sample['f_img'].shape}")
#             results = net(batch_sample)
#             b = time.perf_counter()
#             runtimes.append(b - a)
#             print(f"batch_size: {batch_size}, runtime: {b - a:.2f} seconds")
#             batch_size *= 2
#             del batch_sample
#             del results
#             torch.cuda.empty_cache()
#             gc.collect()

#     except Exception as e:
#         print(e)
#         pass
    
#     print(runtimes)



def add_arguments(subparser):
    subparser.add_argument("--gpu", help="GPU ID. Default: 0", default=0)
    subparser.add_argument("--checkpoint", type=os.path.abspath, help="Path to pretrained models. During training, continue from this checkpoint. During evaluation, evaluate this checkpoint.")
    subparser.add_argument("--params", type=os.path.abspath, help="Path to params.json for hyper-parameters. If a checkpoint is provided, defaults to params.json in the checkpoint folder.")
    subparser.add_argument("--eval_data_configs", type=os.path.abspath, help="Path to data_configs.json for evaluation data information", default='')
    subparser.add_argument("--cudnn", action='store_true', default=False, help="Enable CUDNN for potential speedup")
    subparser.add_argument("--output_dir", type=os.path.abspath, help="Output directory (default: ./vfa)", default='./vfa')

def main():
    logger.info('Program started')

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func='train')
    train_parser.add_argument("--identifier", help="A string that identify the current run (required)", required=True)
    train_parser.add_argument("--use_fused", action='store_true', default=False, help="Use fused kernels")
    train_parser.add_argument("--train_data_configs", type=os.path.abspath, help="Path to data_configs.json for training data information (required)", required=True)
    train_parser.add_argument("--no-sync", action='store_false', dest='sync', default=True, help="Synchronize CUDA operations for accurate timing")
    train_parser.set_defaults(save_results=0)
    add_arguments(train_parser)

    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.set_defaults(func='evaluate')
    evaluate_parser.add_argument("--model_complexity", action='store_true', help="Report computational complexity of the model")
    evaluate_parser.add_argument("--save_results", type=int, default=1, help='''Specify level of evaluation results to save:
        0: no results saved
        1: save minimal outputs
        2: save all inputs and outputs''')
    evaluate_parser.add_argument("--f_img", type=os.path.abspath, help="Path to fixed image (if eval_data_configs not provided)")
    evaluate_parser.add_argument("--m_img", type=os.path.abspath, help="Path to moving image (if eval_data_configs not provided)")
    evaluate_parser.add_argument("--f_input", type=os.path.abspath, help="Path to fixed input image (if eval_data_configs not provided)")
    evaluate_parser.add_argument("--m_input", type=os.path.abspath, help="Path to moving input image (if eval_data_configs not provided)")
    evaluate_parser.add_argument("--f_mask", type=os.path.abspath, help="Path to fixed mask (if eval_data_configs not provided)")
    evaluate_parser.add_argument("--m_mask", type=os.path.abspath, help="Path to moving mask (if eval_data_configs not provided)")
    evaluate_parser.add_argument("--f_seg", type=os.path.abspath, help="Path to fixed label map (if eval_data_configs not provided)")
    evaluate_parser.add_argument("--m_seg", type=os.path.abspath, help="Path to moving label map (if eval_data_configs not provided)")
    evaluate_parser.add_argument("--prefix", type=os.path.abspath, help="Prefix for saved results (if eval_data_configs not provided)")
    evaluate_parser.add_argument("--resolution", type=str, help="Resolution of the data (if eval_data_configs not provided)", choices=['quarter', 'half', 'full'], required=False, default=None)
    evaluate_parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate", required=False, default=None)
    add_arguments(evaluate_parser)

    args = parser.parse_args()

    num_samples = args.num_samples
    print(f"num_samples: {num_samples}")

    '''load hyper parameters'''
    params = {}
    
    # Set is_dummy flag based on identifier
    params['is_dummy'] = 'dummy' in args.identifier.lower() if hasattr(args, 'identifier') else True
    
    if args.checkpoint:
        # if use pretrained model is specified, load the previous hyper paremeters
        params_path = os.path.join(os.path.dirname(args.checkpoint), 'params.json')
        if os.path.exists(params_path):
            params = update_params_json(params_path, params)
        else:
            logger.warning(f'Cannot find params.json for the checkpoint at default path {params_path}')
    if args.params:
        # load the hyper parameters from json file
        params = update_params_json(args.params, params)
    params = update_params_args(args, params)

    # Add use_fused to NCC loss params if provided
    if hasattr(args, 'use_fused') and args.use_fused:
        if 'loss' in params:
            for phase in ['train', 'eval']:
                if phase in params['loss'] and 'NCC' in params['loss'][phase]:
                    params['loss'][phase]['NCC']['use_fused'] = True
                if phase in params['loss'] and 'Affine_NCC' in params['loss'][phase]:
                    params['loss'][phase]['Affine_NCC']['use_fused'] = True
                if phase in params['loss'] and 'MI' in params['loss'][phase]:
                    params['loss'][phase]['MI']['use_fused'] = True
                if phase in params['loss'] and 'Affine_MI' in params['loss'][phase]:
                    params['loss'][phase]['Affine_MI']['use_fused'] = True

    eval_data_configs = load_data_configs(params['eval_data_configs'])
    params['model']['in_channels'] = eval_data_configs['shape'][0]
    params['model']['in_shape'] = eval_data_configs['shape'][1:]

    logger.info('List parameters')
    for item in params:
        logger.info(f'---- {item}:{params[item]}')

    os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu']
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    if params['cudnn']:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        cudnn.deterministic = True

    logger.info('Load datasets')
    kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': False}
    eval_dataset_class = load_dataset(eval_data_configs['loader'])
    evalDS = eval_dataset_class(eval_data_configs, params)
    evalDL = torch.utils.data.DataLoader(evalDS, batch_size=1, shuffle=False, **kwargs)
    # modify shape
    f_img = evalDS[0]['f_img']
    shape = f_img.shape[1:]
    eval_data_configs['shape'] = [1, *shape]
    params['model']['in_shape'] = eval_data_configs['shape'][1:]

    if params['func'] == 'train':
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': True}
        train_data_configs = load_data_configs(params['train_data_configs'])

        train_dataset_class = load_dataset(train_data_configs['loader'])
        trainDS = train_dataset_class(train_data_configs, params)
        trainDL = torch.utils.data.DataLoader(
                                trainDS,
                                batch_size=params['batch_size'],
                                shuffle=True,
                                **kwargs,
        )

    logger.info('Initialize network')
    net = create_network_class(params['model']['name'])(params, device)
    optimizer = optim.Adam(net.parameters(), lr=params['LR'], amsgrad=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['LR_schedule'], gamma=0.5)
    scaler = GradScaler() if params['fp16'] else None

    '''load pretrained model'''
    if params['checkpoint']:
        if os.path.exists(params['checkpoint']): 
            checkpoint = torch.load(params['checkpoint'], map_location=device)
        else:
            raise FileNotFoundError(f"Checkpoint file {params['checkpoint']} does not exist.")
        try:
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            logger.info(f"Load checkpoint {params['checkpoint']}")
        except KeyError:
            '''load pretrained model in other format'''
            start_epoch = net.load_pretrained_model(checkpoint, optimizer)
    else:
        start_epoch = 1

    if params['func'] == 'train':
        logger.info('Training started')
        # setting up tensorboard
        from torch.utils.tensorboard import SummaryWriter
        tb_folder = os.path.join(params['output_dir'], 'runs')
        tb_filename = params['identifier']+'-'+str(datetime.now())
        writer = SummaryWriter(os.path.join(tb_folder, tb_filename))
        logger.info(f'Tensorboard directory: {tb_folder}')
        
        # Initialize wandb if not a dummy run
        if not params['is_dummy']:
            wandb.init(
                project="vfa",
                name=params['identifier'],
                config=params,
                dir=params['output_dir']
            )
            logger.info(f'Wandb initialized for run: {params["identifier"]}')

        # checkpoints
        checkpoint_folder = f"{params['output_dir']}/checkpoints/{params['identifier']}"
        pathlib.Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(checkpoint_folder, 'params.json'), 'w') as f:
            json.dump(params, f, indent=4, separators=(',',':'))
        logger.info(f'Checkpoint directory: {checkpoint_folder}')

        # main progress bar
        main_pbar = tqdm(total=params['num_epochs'], position=0)
        main_pbar.set_description(f"[Machine: {socket.gethostname().split('.')[0]}-{params['gpu']}][Total Epochs: {params['num_epochs']}]")
        eval_pbar = tqdm(total=len(evalDL), position=1)
        train_pbar = tqdm(total=len(trainDS) // params['batch_size'], position=2)

        for epoch in range(start_epoch, params['num_epochs']):

            '''training'''
            train_pbar.reset()
            train_pbar.set_description(f"[Training][Epoch: {epoch}|{params['num_epochs']}]")
            train_pbar.refresh()
            
            # Initialize timer for training
            epoch_loss = train(net, trainDL, params, optimizer, scheduler, scaler, train_pbar, train_data_configs)

            for key in epoch_loss:
                average_loss = np.mean(np.array(epoch_loss[key]))
                writer.add_scalar(f"train/{key}-Loss", average_loss, epoch)
                if not params['is_dummy']:
                    wandb.log({f"train/{key}-Loss": average_loss}, step=epoch)

            '''run validation'''
            eval_pbar.reset()
            eval_pbar.set_description(f"[Evaluation][Epoch: {epoch}|{params['num_epochs']}]")
            eval_pbar.refresh()
            
            # Initialize timer for validation
            start_time = time.time()
            epoch_loss = evaluate(net, evalDL, params, eval_pbar, eval_data_configs, num_samples)
            end_time = time.time()
            logger.info(f"Validation time: {end_time - start_time:.2f} seconds")

            for key in epoch_loss:
                average_loss = np.mean(np.array(epoch_loss[key]))
                writer.add_scalar(f"validate/{key}-Loss", average_loss, epoch)
                if not params['is_dummy']:
                    wandb.log({f"validate/{key}-Loss": average_loss}, step=epoch)

            '''save model'''
            save_model_name = str(epoch) + '-net.pth' if epoch % 50 == 0 else 'most_recent.pth'
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(checkpoint_folder, save_model_name))


            main_pbar.update(1)
        
        # Close wandb if it was initialized
        if not params['is_dummy']:
            wandb.finish()

    elif params['func'] == 'evaluate':
        params['resolution'] = args.resolution
        if params['model_complexity']:
            logger.info('Analyze FLOPs and Number of Parameters')
            # net.print_num_parameters()
            net.print_flops()

        logger.info('Evaluation started')
        pbar = tqdm(total=len(evalDS), position=0)
        pbar.set_description(f"[Evaluation]")
        
        # Initialize timer for evaluation
        epoch_loss = evaluate(net, evalDL, params, pbar, eval_data_configs, num_samples)
        pbar.close()

        # save dice scores
        dice_save_path = os.path.join(params['prefix'], 'dice_scores.npy')
        os.makedirs(os.path.dirname(dice_save_path), exist_ok=True)
        np.save(dice_save_path, np.stack(epoch_loss['Dice']))
        logger.info(f"Save dice scores to {dice_save_path}")

        for key in epoch_loss:
            loss = np.mean(np.array(epoch_loss[key]))
            print(f"{key}-Loss --- mean: {np.mean(loss):.4f} std: {np.std(loss):.4f}")

        # xlsx_path = os.path.join(str(result_path), f"{params['output']}_reg_statistics.xlsx")
        # save_results_to_xlsx(epoch_loss, xlsx_path)
        # logger.info(f"Save statistics to {xlsx_path}")

if __name__ == '__main__':
    main()
