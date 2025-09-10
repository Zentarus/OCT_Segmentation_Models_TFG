# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np

def pad_image_top(img, target_height=512, fill_value=0):
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    pad_amount = target_height - h
    if pad_amount <= 0:
        return img

    if img_np.ndim == 2:
        pad_array = np.full((pad_amount, w), fill_value, dtype=img_np.dtype)
    else:
        pad_array = np.full((pad_amount, w, img_np.shape[2]), fill_value, dtype=img_np.dtype)

    padded = np.vstack((pad_array, img_np))
    return Image.fromarray(padded)

def process_dataset(src_root, dst_root):
    splits = ['train', 'eval', 'test']
    for split in splits:
        for data_type in ['img', 'mask']:
            src_dir = os.path.join(src_root, split, data_type)
            dst_dir = os.path.join(dst_root, split, data_type)
            os.makedirs(dst_dir, exist_ok=True)

            file_list = os.listdir(src_dir)
            total_files = len(file_list)

            for idx, fname in enumerate(file_list):
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dst_dir, fname)

                img = Image.open(src_path)

                padded_img = pad_image_top(img, target_height=512, fill_value=0)
                padded_img.save(dst_path)

                percent = (idx + 1) / total_files * 100
                print(f'Procesando {split}/{data_type}: {percent:.1f}% ({idx + 1}/{total_files})', end='\r')
            print()  

if __name__ == "__main__":
    src_dataset = "/IronWolf/oct_seg/oct_MGUNet/oct_JH/data/Dataset_JH_2D"
    dst_dataset = "/IronWolf/oct_seg/oct_MGUNet/oct_JH/data/Dataset_JH_2D_preprocessed"
    process_dataset(src_dataset, dst_dataset)
