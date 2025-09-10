#!/home/users/user1/anaconda3/envs/env_py38/bin/python
# -*- coding: utf-8 -*-
# @Author: Jiaxuan Li
##### System library #####
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os.path as osp
from os.path import exists
import argparse
import json
import logging
import time
import copy
##### pytorch library #####
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
##### My own library #####
import data.seg_transforms as dt
from data.seg_dataset import segList
from utils.logger import Logger
from utils.augmentation import AlbumentationsTransform  # Nueva importación
from models.net_builder import net_builder
from utils.loss import loss_builder1, loss_builder2
from utils.utils import adjust_learning_rate, AverageMeter, save_model
from utils.utils import compute_dice, compute_pa, compute_single_avg_score
from utils.vis import vis_result
#### Training logger ####
from utils.training_logger import TrainingLogger

# logger vis
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger_vis = logging.getLogger(__name__)
logger_vis.setLevel(logging.DEBUG)

# training process
def train(args, train_loader, model, criterion1, criterion2, optimizer, epoch, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    dice = AverageMeter()
    dice_meters = [AverageMeter() for _ in range(1, args.num_classes)]
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input_var = Variable(input).cuda()
        target_var_seg = Variable(target).cuda()
        target_var_seg1 = copy.deepcopy(target_var_seg)
        input_var1 = copy.deepcopy(input_var)
        target_var_seg1[target_var_seg == 0] = 0
        for cls in range(1, args.num_classes):
            target_var_seg1[target_var_seg == cls] = 1
        target_var_seg1[target_var_seg == args.num_classes] = 2
        output_seg1, _, output_seg = model(input_var1)
        loss_1_1 = criterion1[0](output_seg1, target_var_seg1)
        loss_1_2 = criterion1[1](output_seg1, target_var_seg1)
        loss_1 = loss_1_1 + loss_1_2
        loss_2_1 = criterion2[0](output_seg, target_var_seg)
        loss_2_2 = criterion2[1](output_seg, target_var_seg)
        loss_2 = loss_2_1 + loss_2_2
        loss = loss_1 + 2 * loss_2
        losses.update(loss.data, input.size(0))
        _, pred_seg = torch.max(output_seg, 1)
        pred_seg = pred_seg.cpu().data.numpy()
        label_seg = target_var_seg.cpu().data.numpy()
        ret_d = compute_dice(label_seg, pred_seg, args.num_classes)
        dice_score = compute_single_avg_score(ret_d, args.num_classes)
        dice.update(dice_score)
        for idx, meter in enumerate(dice_meters, 1):
            meter.update(ret_d[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            log_str = f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tDice {dice.val:.4f} ({dice.avg:.4f})\t'
            for idx, meter in enumerate(dice_meters, 1):
                log_str += f'Dice_{idx} {meter.val:.4f} ({meter.avg:.4f})\t'
            logger_vis.info(log_str)
            print('Loss :', loss.cpu().data.numpy())
    return [losses.avg, dice.avg] + [meter.avg for meter in dice_meters]

# evaluation process
def eval(phase, args, eval_data_loader, model, result_path=None, logger=None):
    batch_time = AverageMeter()
    dice = AverageMeter()
    mpa = AverageMeter()
    dice_meters = [AverageMeter() for _ in range(1, args.num_classes)]
    pa_meters = [AverageMeter() for _ in range(1, args.num_classes)]
    dice_list, mpa_list = [], []
    ret_dice, ret_pa = [], []
    model.eval()
    end = time.time()
    pred_seg_batch = []
    label_seg_batch = []
    for iter, (image, label, imt, imn) in enumerate(eval_data_loader):
        with torch.no_grad():
            image_var = Variable(image).cuda()
            _, _, output_seg = model(image_var)
            _, pred_seg = torch.max(output_seg, 1)
            pred_seg = pred_seg.cpu().data.numpy().astype('uint8')
            if phase == 'eval' or phase == 'test':
                imt = (imt.squeeze().numpy()).astype('uint8')
                ant = label.numpy().astype('uint8')
                save_dir = osp.join(result_path, 'vis')
                if not exists(save_dir): os.makedirs(save_dir)
                if not exists(save_dir + '/label'): os.makedirs(save_dir + '/label')
                if not exists(save_dir + '/pred'): os.makedirs(save_dir + '/pred')
                vis_result(imn, imt, ant, pred_seg, save_dir, args.num_classes)
                print('Saved visualized results!')
            label_seg = label.numpy().astype('uint8')
            pred_seg_batch.append(pred_seg)
            label_seg_batch.append(label_seg)
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
        batch_time.update(time.time() - end)
        end = time.time()
        log_str = f'{phase.upper()}: [{iter}/{len(eval_data_loader)}]\tID {imn[0].split(".")[0]}\tDice {dice.val:.4f}\t'
        for idx, (d_meter, p_meter) in enumerate(zip(dice_meters, pa_meters), 1):
            log_str += f'Dice_{idx} {d_meter.val:.4f}\tPA_{idx} {p_meter.val:.4f}\t'
        log_str += f'MPA {mpa.val:.4f}\tBatch_time {batch_time.val:.3f}\t'
        logger_vis.info(log_str)
    final_dice_avg = dice.avg
    final_mpa_avg = mpa.avg
    final_dice_scores = [meter.avg for meter in dice_meters]
    final_pa_scores = [meter.avg for meter in pa_meters]
    print('######  Segmentation Result  ######')
    print(f'Final Dice_avg Score: {final_dice_avg:.4f}')
    for idx, score in enumerate(final_dice_scores, 1):
        print(f'Final Dice_{idx} Score: {score:.4f}')
    print(f'Final PA_avg: {final_mpa_avg:.4f}')
    for idx, score in enumerate(final_pa_scores, 1):
        print(f'Final PA_{idx} Score: {score:.4f}')
    if phase == 'eval' or phase == 'test':
        logger.append([final_dice_avg] + final_dice_scores + [final_mpa_avg] + final_pa_scores)
    return [final_dice_avg] + final_dice_scores + [dice_list]

##### Para logger de train ######
def append_metrics_to_json(file_path, metrics_dict):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = []
    data.append(metrics_dict)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

###### train ######
def train_seg(args, train_result_path, train_loader, eval_loader):
    metrics_path = osp.join(train_result_path, 'metrics_history.json')
    for k, v in args.__dict__.items():
        print(k, ':', v)
    net = net_builder(args.name, n_classes=args.num_classes)
    model = torch.nn.DataParallel(net).cuda()
    print('#' * 15, args.name, '#' * 15)
    criterion1 = loss_builder1()
    criterion2 = loss_builder2(args.num_classes)
    optimizer = torch.optim.Adam(
        net.parameters(),
        args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay
    )
    cudnn.benchmark = True
    best_dice = 0
    logger_nn = TrainingLogger(verbose=True)
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        logger_nn.log('epoch_start_timestamps', start_time, epoch)
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger_nn.log('lrs', lr, epoch)
        logger_vis.info(f'Epoch: [{epoch}]\t')
        train_metrics = train(args, train_loader, model, criterion1, criterion2, optimizer, epoch)
        loss, dice_train = train_metrics[0], train_metrics[1]
        dice_train_classes = train_metrics[2:]
        eval_metrics = eval('train', args, eval_loader, model)
        dice_val = eval_metrics[0]
        dice_val_classes = eval_metrics[1:-1]
        dice_list = eval_metrics[-1]
        logger_nn.log('train_losses', float(loss), epoch)
        logger_nn.log('val_losses', float(1 - dice_val), epoch)
        logger_nn.log('mean_fg_dice', float(dice_val), epoch)
        end_time = time.time()
        logger_nn.log('epoch_end_timestamps', end_time, epoch)
        is_best = dice_val > best_dice
        best_dice = max(dice_val, best_dice)
        model_dir = osp.join(train_result_path, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'dice_epoch': dice_val,
            'best_dice': best_dice,
        }, is_best, model_dir)
        metrics_entry = {
            "epoch": epoch,
            "train_loss": float(loss),
            "val_loss": float(1 - dice_val),
            "mean_fg_dice": float(dice_val),
            "dice_train": float(dice_train),
            "dice_val": float(dice_val),
            "dice_classes": {
                **{f"class{i}_train": float(dice_train_classes[i-1]) for i in range(1, args.num_classes)},
                **{f"class{i}_val": float(dice_val_classes[i-1]) for i in range(1, args.num_classes)}
            },
            "learning_rate": float(lr),
            "epoch_duration_sec": float(end_time - start_time)
        }
        append_metrics_to_json(metrics_path, metrics_entry)
        logger_nn.plot_progress_png(train_result_path)

###### validation ######
def eval_seg(args, eval_result_path, eval_loader):
    logger_eval = Logger(osp.join(eval_result_path, 'dice_mpa_epoch.txt'), title='dice&mpa', resume=False)
    logger_eval.set_names(['Dice'] + [f'Dice_{i}' for i in range(1, args.num_classes)] + ['mpa'] + [f'pa_{i}' for i in range(1, args.num_classes)])
    print('Loading eval model: {}'.format(args.name))
    net = net_builder(args.name, n_classes=args.num_classes)
    model = torch.nn.DataParallel(net).cuda()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded!')
    cudnn.benchmark = True
    eval('eval', args, eval_loader, model, eval_result_path, logger_eval)

###### test ######
def test_seg(args, test_result_path, test_loader):
    logger_test = Logger(osp.join(test_result_path, 'dice_mpa_epoch.txt'), title='dice&mpa', resume=False)
    logger_test.set_names(['Dice'] + [f'Dice_{i}' for i in range(1, args.num_classes)] + ['mpa'] + [f'pa_{i}' for i in range(1, args.num_classes)])
    print('Loading test model ...')
    net = net_builder(args.name, n_classes=args.num_classes)
    model = torch.nn.DataParallel(net).cuda()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded!')
    cudnn.benchmark = True
    eval('test', args, test_loader, model, test_result_path, logger_test)

def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('--name', dest='name', help='change model', default=None, type=str)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--step', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--t', type=str, default='t1')
    parser.add_argument('--model-path', help='pretrained model test', default=' ', type=str)
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Numero de clases para segmentar (default: 10)')
    parser.add_argument('--class_mapping', type=str, default=None,
                        help='JSON string con mapeo de clases, ej 3 clases (RNFL, GCL-RPE, BACKGROUND): \'{"1":1,"2-8":2,"9":0}\'')
    parser.add_argument('--augment', action='store_true', help='Activar data augmentation similar a nnU-Net')
    args = parser.parse_args()
    return args

def main():
    ##### config #####
    args = parse_args()
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('torch version:', torch.__version__)
    ##### result path setting #####
    tn = args.t
    task_name = args.data_dir.split('/')[-2] + '/' + args.data_dir.split('/')[-1]
    train_result_path = osp.join('result', task_name, 'train', args.name + '_' + str(args.lr) + '_' + tn)
    if not exists(train_result_path):
        os.makedirs(train_result_path)
    test_result_path = osp.join('result', task_name, 'test', args.name + '_' + str(args.lr) + '_' + tn)
    if not exists(test_result_path):
        os.makedirs(test_result_path)
    ##### load dataset #####
    info = json.load(open(osp.join(args.data_dir, 'info.json'), 'r'))
    normalize = dt.Normalize(mean=info['mean'], std=info['std'])
    class_mapping = args.class_mapping if hasattr(args, 'class_mapping') and args.class_mapping else None
    # Transforms base
    base_transforms = [
        dt.Label_Transform(class_mapping=class_mapping),
    ]
    # Agregar augmentation solo si --augment
    if args.augment:
        base_transforms.append(AlbumentationsTransform(augment=True, target_height=512, target_width=1024))
    # Transforms finales
    base_transforms += [
        dt.ToTensor(),
        normalize
    ]
    train_dataset = segList(args.data_dir, 'train', dt.Compose(base_transforms))
    val_test_transforms = [
        dt.Label_Transform(class_mapping=class_mapping),
        dt.ToTensor(),
        normalize
    ]
    val_dataset = segList(args.data_dir, 'eval', dt.Compose(val_test_transforms))
    test_dataset = segList(args.data_dir, 'test', dt.Compose(val_test_transforms))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True
    )
    eval_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False
    )
    ##### train #####
    train_seg(args, train_result_path, train_loader, eval_loader)
    ##### test #####
    model_best_path = osp.join(train_result_path, 'model', 'model_best.pth.tar')
    args.model_path = model_best_path
    test_seg(args, test_result_path, test_loader)

if __name__ == '__main__':
    main()