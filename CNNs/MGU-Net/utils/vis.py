import numpy as np
import os
import os.path as osp
import cv2

def vis_result(imn, imt, ant, pred, save_dir, n_class=10):
    # Handle imn as list, tuple, or string
    if isinstance(imn, (list, tuple)):
        imn = imn[0]  # Take first element if list or tuple
    if not isinstance(imn, str):
        raise ValueError(f'Expected imn to be a string, got {type(imn)}: {imn}')
    
    # Squeeze extra dimensions from inputs
    imt = imt.squeeze()  # Remove batch/channel dimensions
    pred = pred.squeeze()  # Remove batch/channel dimensions
    ant = ant.squeeze() if ant is not None else None
    
    img = gray2rgbimage(imt)
    pred_img = draw_img(imt, pred, n_class=n_class)
    if ant is None:
        # Save combined image (original + colored prediction) in vis
        cv2.imwrite(osp.join(save_dir, imn), np.hstack((img, pred_img)).astype('uint8'))
        # Save grayscale prediction mask in pred, scaled to [0, 255]
        pred_scaled = (pred * (255 / (n_class - 1))).astype('uint8')  # Scale [0, 9] to [0, 255]
        cv2.imwrite(osp.join(save_dir, 'pred', imn), pred_scaled)
    else:
        ant_img = draw_img(imt, ant, n_class=n_class)
        # Save combined image (original + ground truth + colored prediction) in vis
        cv2.imwrite(osp.join(save_dir, imn), np.hstack((img, ant_img, pred_img)).astype('uint8'))
        # Save ground truth in label, scaled to [0, 255]
        ant_scaled = (ant * (255 / (n_class - 1))).astype('uint8')  # Scale [0, 9] to [0, 255]
        cv2.imwrite(osp.join(save_dir, 'label', imn), ant_scaled)
        # Save grayscale prediction mask in pred, scaled to [0, 255]
        pred_scaled = (pred * (255 / (n_class - 1))).astype('uint8')  # Scale [0, 9] to [0, 255]
        cv2.imwrite(osp.join(save_dir, 'pred', imn), pred_scaled)

def draw_img(img, seg, title=None, n_class=10):
    # Ensure inputs are 2D
    img = img.squeeze() if img.ndim > 2 else img
    seg = seg.squeeze() if seg.ndim > 2 else seg
    
    mask = img.copy()  # Copy input image
    label_set = [i + 1 for i in range(n_class)]  # Classes 1 to n_class
    
    # Define colors for 10 classes
    color_set = {
        1: (255, 0, 0),    # NFL
        2: (0, 255, 0),    # GCL
        3: (0, 0, 255),    # IPL
        4: (0, 255, 255),  # INL
        5: (255, 0, 255),  # OPL
        6: (255, 255, 0),  # ONL
        7: (0, 0, 150),    # IS/OS
        8: (0, 150, 0),    # RPE
        9: (150, 0, 150),  # Choroid
        10: (100, 50, 250) # Last class
    }
    
    mask = gray2rgbimage(mask)
    img = gray2rgbimage(img)
    if title is not None:
        mask = cv2.putText(mask, title, (16, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
    
    # Apply colors to segmentation mask
    for draw_label in label_set:
        if draw_label in color_set:
            mask[seg == draw_label] = color_set[draw_label]
    
    img_mask = cv2.addWeighted(img, 0.4, mask, 0.6, 0)
    return img_mask

def gray2rgbimage(image):
    # Squeeze extra dimensions (e.g., batch or channel)
    image = image.squeeze() if image.ndim > 2 else image
    if image.ndim != 2:
        raise ValueError(f'Expected 2D array for gray2rgbimage, got shape {image.shape}')
    
    height, width = image.shape
    new_img = np.ones((height, width, 3), dtype=np.uint8)
    new_img[:, :, 0] = image.astype('uint8')
    new_img[:, :, 1] = image.astype('uint8')
    new_img[:, :, 2] = image.astype('uint8')
    return new_img