import numbers
import random
import json
import numpy as np
from PIL import Image, ImageOps
import torch

class Resize(object):
    def __init__(self, size=(512, 1024)):  # Size as (height, width) to match training (1024x512)
        self.size = size

    def __call__(self, image, label=None):
        # Resize image to target size
        image = image.resize(self.size[::-1], Image.BILINEAR)  # PIL uses (width, height)
        if label is None:
            return image, None
        return image, label

class Label_Transform(object):
    def __init__(self, label_pixel=(26, 51, 77, 102, 128, 153, 179, 204, 230, 255), class_mapping=None):
        self.label_pixel = label_pixel  # Original pixel values
        self.class_mapping = class_mapping  # Dictionary for class grouping

    def __call__(self, image, label=None, *args):
        if label is None:
            return image, None  # Return only image for inference

        label = np.array(label)
        # Map original pixel values to consecutive labels (1 to 9, background=0)
        for i in range(len(self.label_pixel)):
            label[label == self.label_pixel[i]] = i + 1

        # Apply class mapping if provided
        if self.class_mapping is not None:
            mapping = json.loads(self.class_mapping) if isinstance(self.class_mapping, str) else self.class_mapping
            new_label = np.zeros_like(label)
            for key, new_val in mapping.items():
                if '-' in key:
                    start, end = map(int, key.split('-'))
                    mask = (label >= start) & (label <= end)
                else:
                    mask = (label == int(key))
                new_label[mask] = new_val
            label = new_label

        return image, Image.fromarray(label.astype(np.uint8))

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor([mean] if isinstance(mean, (int, float)) else mean)
        self.std = torch.FloatTensor([std] if isinstance(std, (int, float)) else std)

    def __call__(self, image, label=None):
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        if label is None:
            return image,
        else:
            return image, label

class ToTensor(object):
    def __call__(self, pic, label=None):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(np.copy(pic))  # Ensure writable array
        else:
            # Convert PIL Image to grayscale if not already
            if pic.mode != 'L':
                pic = pic.convert('L')  # Convert to grayscale (1 channel)
            # Convert PIL Image to tensor
            img_data = np.frombuffer(pic.tobytes(), dtype=np.uint8)
            img_data = np.copy(img_data)  # Ensure writable array
            img = torch.from_numpy(img_data)
            nchannel = 1  # Grayscale images have 1 channel
            img = img.view(pic.size[1], pic.size[0], nchannel)
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255)
        if label is None:
            return img,
        else:
            return img, torch.LongTensor(np.array(label, dtype=int))

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args