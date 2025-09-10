import numpy as np
import torch
import os
import os.path as osp
import cv2
import scipy.misc as misc
import shutil
from skimage import measure
import math
import traceback
from sklearn import metrics
import zipfile


def adjust_learning_rate(args, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(state, is_best, model_path):
    model_latest_path = osp.join(model_path, 'model_latest.pth.tar')   
    torch.save(state, model_latest_path)
    if is_best:
        model_best_path = osp.join(model_path, 'model_best.pth.tar')
        shutil.copyfile(model_latest_path, model_best_path)


def save_dice_single(is_best, filename='dice_single.txt'):
    if is_best:
        shutil.copyfile(filename, 'dice_best.txt')


def compute_dice(ground_truth, prediction, class_num=10):  
    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    try:
        ret = [0.5] * class_num
        for i in range(class_num): 
            mask1 = (ground_truth == i)
            mask2 = (prediction == i)
            if mask1.sum() != 0:
                ret[i] = float(2 * ((mask1 * (ground_truth == prediction)).sum()) / (mask1.sum() + mask2.sum()))
            else:
                ret[i] = float('nan')
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret


def compute_pa(ground_truth, prediction, class_num=10): 
    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    try:
        ret = [0.5] * class_num
        for i in range(class_num): 
            mask1 = (ground_truth == i)
            if mask1.sum() != 0:
                ret[i] = float(((mask1 * (ground_truth == prediction)).sum()) / (mask1.sum()))
            else:
                ret[i] = float('nan')
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret


def compute_avg_score(ret_seg, num_classes=10):
    sums = [0.0] * num_classes
    counts = [0.0] * num_classes

    num = len(ret_seg)
    for i in range(num):
        for c in range(num_classes):
            if not math.isnan(ret_seg[i][c]):
                sums[c] += ret_seg[i][c]
                counts[c] += 1

    avgs = [sums[c] / max(counts[c], 1) for c in range(num_classes)]

    # excluimos la clase 0 (background) en el calculo del promedio global
    valid_avgs = avgs[1:]  
    avg_seg = sum(valid_avgs) / len(valid_avgs) if valid_avgs else float('nan')

    return avg_seg, *avgs


def compute_single_avg_score(ret_seg, num_classes=10):
    # excluimos la clase 0 en el calculo del promedio
    class_scores = []
    for c in range(1, num_classes):
        class_scores.append(ret_seg[c] if not math.isnan(ret_seg[c]) else 0.0)

    avg_seg = sum(class_scores) / (num_classes - 1) if num_classes > 1 else float('nan')
    return avg_seg

