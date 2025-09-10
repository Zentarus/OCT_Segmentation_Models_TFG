import os
from PIL import Image
import numpy as np
import json

def calculate_mean_std(image_dir):
    means = []
    stds = []
    channel_count = None

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path)
        
        # detectamos numero de canales
        if channel_count is None:
            if img.mode == 'L':  
                channel_count = 1
            elif img.mode == 'RGB':
                channel_count = 3
            else:
                channel_count = len(img.getbands())  # caso general
            
            print(f"Numero de canales detectado: {channel_count}")

        img_np = np.array(img).astype(np.float32) / 255.0  # normalizamos 0-1
        
        # si es escala de grises, expandimos dims
        if channel_count == 1:
            img_np = np.expand_dims(img_np, axis=2)

        means.append(np.mean(img_np, axis=(0,1)))
        stds.append(np.std(img_np, axis=(0,1)))
    
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    train_img_dir = '/IronWolf/oct_seg/oct_MGUNet/oct_JH/data/Dataset_JH_2D_preprocessed/train/img/'
    dataset_root = '/IronWolf/oct_seg/oct_MGUNet/oct_JH/data/Dataset_JH_2D_preprocessed/'

    mean, std = calculate_mean_std(train_img_dir)
    print('Mean:', mean)
    print('Std:', std)

    info = {
        "mean": mean,
        "std": std
    }

    info_path = os.path.join(dataset_root, 'info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    print(f'Archivo info.json guardado en: {info_path}')
