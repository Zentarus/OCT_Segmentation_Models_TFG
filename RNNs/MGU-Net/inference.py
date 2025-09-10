#!/home/users/user1/anaconda3/envs/env_py38/bin/python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os.path as osp
from os.path import exists
import argparse
import json
import logging
import time
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import data.seg_transforms as dt
from data.seg_dataset import segList
from utils.logger import Logger
from models.net_builder import net_builder
from utils.utils import compute_dice, compute_pa, compute_single_avg_score, AverageMeter
from utils.vis import vis_result
from tqdm import tqdm

# Logger setup
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger_vis = logging.getLogger(__name__)
logger_vis.setLevel(logging.DEBUG)

def eval(phase, args, eval_data_loader, model, result_path, logger):
    model.eval()
    end = time.time()
    
    for iter, data in tqdm(enumerate(eval_data_loader), total=len(eval_data_loader), desc="Inference Progress"):
        if args.inference_only or phase == 'predict':
            image, imt, imn = data
            label = None
        else:
            image, label, imt, imn = data
        
        with torch.no_grad():
            image_var = Variable(image).cuda()
            _, _, output_seg = model(image_var)
            _, pred_seg = torch.max(output_seg, 1)
            pred_seg = pred_seg.cpu().data.numpy().astype('uint8')
            
            save_dir = osp.join(result_path, 'vis')
            if not exists(save_dir):
                os.makedirs(save_dir)
            if not exists(save_dir + '/pred'):
                os.makedirs(save_dir + '/pred')
            if label is not None and not args.inference_only:
                if not exists(save_dir + '/label'):
                    os.makedirs(save_dir + '/label')
            
            imt = (imt.squeeze().numpy()).astype('uint8')
            ant = label.numpy().astype('uint8') if label is not None else None
            vis_result(imn, imt, ant, pred_seg, save_dir, args.num_classes)
            
            log_str = f'{phase.upper()}: [{iter}/{len(eval_data_loader)}] ID {imn[0].split(".")[0]}'
            logger_vis.info(log_str)
    
    return None

def eval_with_metrics(phase, args, eval_data_loader, model, result_path, logger):
    batch_time = AverageMeter()
    dice = AverageMeter()
    mpa = AverageMeter()
    dice_meters = [AverageMeter() for _ in range(1, args.num_classes)]
    pa_meters = [AverageMeter() for _ in range(1, args.num_classes)]
    dice_list, mpa_list = [], []
    ret_dice, ret_pa = [], []

    per_sample_results = [] if args.save_per_sample else None

    model.eval()
    end = time.time()
    
    for iter, (image, label, imt, imn) in tqdm(enumerate(eval_data_loader), total=len(eval_data_loader), desc="Inference Progress"):
        with torch.no_grad():
            image_var = Variable(image).cuda()
            _, _, output_seg = model(image_var)
            _, pred_seg = torch.max(output_seg, 1)
            pred_seg = pred_seg.cpu().data.numpy().astype('uint8')
            
            save_dir = osp.join(result_path, 'vis')
            if not exists(save_dir):
                os.makedirs(save_dir)
            if not exists(save_dir + '/label'):
                os.makedirs(save_dir + '/label')
            if not exists(save_dir + '/pred'):
                os.makedirs(save_dir + '/pred')
            
            imt = (imt.squeeze().numpy()).astype('uint8')
            ant = label.numpy().astype('uint8')
            vis_result(imn, imt, ant, pred_seg, save_dir, args.num_classes)
            
            label_seg = label.numpy().astype('uint8')
            ret_d = compute_dice(label_seg, pred_seg, args.num_classes)
            ret_p = compute_pa(label_seg, pred_seg, args.num_classes)
            ret_dice.append(ret_d)
            ret_pa.append(ret_p)
            dice_score = compute_single_avg_score(ret_d, args.num_classes)
            mpa_score = compute_single_avg_score(ret_p, args.num_classes)
            dice_list.append(dice_score)
            mpa_list.append(mpa_score)
            dice.update(dice_score)
            mpa.update(mpa_score)
            for idx, (d_meter, p_meter) in enumerate(zip(dice_meters, pa_meters), 1):
                d_meter.update(ret_d[idx])
                p_meter.update(ret_p[idx])

            if args.save_per_sample:
                per_sample_results.append({
                    "file": imn[0],
                    "dice": {f"dice_{i}": float(ret_d[i]) for i in range(1, args.num_classes)},
                    "pa": {f"pa_{i}": float(ret_p[i]) for i in range(1, args.num_classes)}
                })
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            log_str = f'{phase.upper()}: [{iter}/{len(eval_data_loader)}] ID {imn[0].split(".")[0]} Dice {dice.val:.4f} '
            for idx, (d_meter, p_meter) in enumerate(zip(dice_meters, pa_meters), 1):
                log_str += f'Dice_{idx} {d_meter.val:.4f} PA_{idx} {p_meter.val:.4f} '
            log_str += f'MPA {mpa.val:.4f} Batch_time {batch_time.val:.3f}'
            logger_vis.info(log_str)
    
    final_dice_avg = dice.avg
    final_mpa_avg = mpa.avg
    final_dice_scores = [meter.avg for meter in dice_meters]
    final_pa_scores = [meter.avg for meter in pa_meters]

    dice_array = np.array(ret_dice)
    pa_array = np.array(ret_pa)
    dice_std = np.std(dice_array[:,1:], axis=0)
    pa_std = np.std(pa_array[:,1:], axis=0)
    
    logger.append(
        [final_dice_avg] + final_dice_scores + 
        [final_mpa_avg] + final_pa_scores +
        list(dice_std) + list(pa_std)
    )

    if args.save_per_sample:
        json_path = osp.join(result_path, f"{phase}_per_sample_results.json")
        with open(json_path, "w") as f:
            json.dump({
                "per_sample": per_sample_results,
                "dice_avg": float(final_dice_avg),
                "pa_avg": float(final_mpa_avg),
                "dice_std": {f"dice_{i+1}": float(val) for i, val in enumerate(dice_std)},
                "pa_std": {f"pa_{i+1}": float(val) for i, val in enumerate(pa_std)}
            }, f, indent=4)
        print(f"Saved detailed per-sample results to {json_path}")
    
    return [final_dice_avg] + final_dice_scores + [dice_list]

def test_seg(args, test_result_path, test_loader):
    logger_test = Logger(
        osp.join(test_result_path, 'dice_mpa_epoch.txt'),
        title='dice&mpa',
        resume=False
    )

    names = (
        ['Dice'] +
        [f'Dice_{i}' for i in range(1, args.num_classes)] +
        ['mpa'] +
        [f'pa_{i}' for i in range(1, args.num_classes)] +
        [f'dice_std_{i}' for i in range(1, args.num_classes)] +
        [f'pa_std_{i}' for i in range(1, args.num_classes)]
    )

    logger_test.set_names(names)

    net = net_builder(args.name, n_classes=args.num_classes)
    model = torch.nn.DataParallel(net).cuda()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    cudnn.benchmark = True
    
    if args.inference_only:
        eval('predict', args, test_loader, model, test_result_path, logger_test)
    else:
        eval_with_metrics('test', args, test_loader, model, test_result_path, logger_test)

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with MGU-Net for original dataset')
    parser.add_argument('-d', '--data-dir', default=None, required=True, help='Path to dataset directory containing info.json for normalization')
    parser.add_argument('--test-dir', default=None, required=True, help='Path to test data directory containing img subfolder')
    parser.add_argument('--name', default='tsmgunet', help='Model name (default: tsmgunet)')
    parser.add_argument('--model-path', default=None, required=True, help='Path to pretrained model weights')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes to segment (default: 10)')
    parser.add_argument('--class_mapping', type=str, default=None, help='JSON string for class mapping (e.g., {"0": 0, "1": 1, ..., "9": 9})')
    parser.add_argument('-j', '--workers', type=int, default=0, help='Number of data loading workers (default: 0)')
    parser.add_argument('--t', type=str, default=None, required=True, help='Test identifier to append to output directory')
    parser.add_argument('--inference-only', action='store_true', help='Perform inference only without masks and metrics')
    parser.add_argument('--save-per-sample', action='store_true', help='Save per-image metrics and standard deviation in JSON')
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    
    img_dir = osp.join(args.test_dir, 'img')
    if not os.path.exists(img_dir):
        raise ValueError(f"Directory {img_dir} does not exist. Check image directory")
    
    if not args.inference_only:
        mask_dir = osp.join(args.test_dir, 'mask')
        if not os.path.exists(mask_dir):
            raise ValueError(f"Directory {mask_dir} does not exist. Check mask directory or use --inference-only")
    
    task_name = args.test_dir.split('/')[-2] + '/' + args.test_dir.split('/')[-1]
    test_result_path = osp.join('result', task_name, 'test', args.name + '_' + args.t)
    if not exists(test_result_path):
        os.makedirs(test_result_path)
    
    info = json.load(open(osp.join(args.data_dir, 'info.json'), 'r'))
    normalize = dt.Normalize(mean=info['mean'], std=info['std'])
    
    class_mapping = args.class_mapping if args.class_mapping else None
    t = [dt.Label_Transform(class_mapping=class_mapping), dt.ToTensor(), dt.Normalize(mean=info['mean'], std=info['std'])]
    
    phase = 'predict' if args.inference_only else 'eval'
    test_dataset = segList(args.test_dir, phase, dt.Compose(t), inference_only=args.inference_only)
    if len(test_dataset) == 0:
        raise ValueError(f"No images found in {img_dir}. Check directory or file formats")
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False
    )
    
    test_seg(args, test_result_path, test_loader)

if __name__ == '__main__':
    main()