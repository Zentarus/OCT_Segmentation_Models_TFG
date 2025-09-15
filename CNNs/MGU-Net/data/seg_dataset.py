from torch.utils.data import Dataset
from os.path import join, exists
from PIL import Image
import torch
import glob
import os
import os.path as osp
import numpy as np
import torchvision.transforms as tt
import data.seg_transforms as st
import PIL
import random

class segList(Dataset):
    def __init__(self, data_dir, phase, transforms, inference_only=False):
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.inference_only = inference_only
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
      if self.phase == 'predict' and self.inference_only:
          try:
              image = Image.open(self.image_list[index])
              # Convert RGBA to grayscale if necessary
              if image.mode == 'RGBA':
                  image = image.convert('L')  # Convert to grayscale
          except Exception as e:
              raise ValueError(f"Failed to open image {self.image_list[index]}: {e}")
          try:
              # Create imt as a 2D tensor
              imt = torch.from_numpy(np.array(image)).float()  # Shape: (H, W) for grayscale
          except Exception as e:
              raise ValueError(f"Failed to convert image {self.image_list[index]} to tensor: {e}")
          try:
              # Apply transforms to the image
              image = self.transforms(image)
              if image is None:
                  raise ValueError(f"Transforms returned None for image at {self.image_list[index]}")
              # Handle case where transforms return a tuple
              if isinstance(image, tuple):
                  #print(f"Transforms returned a tuple for image at {self.image_list[index]}: {image}")
                  if len(image) != 1:
                      raise ValueError(f"Transforms returned a tuple with unexpected length {len(image)} for image at {self.image_list[index]}")
                  image = image[0]  # Unwrap the tuple if it contains a single tensor
              # Ensure image is a tensor
              if not isinstance(image, torch.Tensor):
                  raise ValueError(f"Transforms did not return a tensor for image at {self.image_list[index]}, got {type(image)}")
          except Exception as e:
              raise ValueError(f"Failed to apply transforms to {self.image_list[index]}: {e}")
          imn = self.image_list[index].split('/')[-1]
          # Return tuple without label
          return (image, imt, imn)
      else:
          try:
              data = [Image.open(self.image_list[index])]
              # Convert RGBA to grayscale if necessary
              if data[0].mode == 'RGBA':
                  data[0] = data[0].convert('L')
          except Exception as e:
              raise ValueError(f"Failed to open image {self.image_list[index]}: {e}")
          try:
              imt = torch.from_numpy(np.array(data[0])).float()  # Shape: (H, W) for grayscale
          except Exception as e:
              raise ValueError(f"Failed to convert image {self.image_list[index]} to tensor: {e}")
          try:
              data.append(Image.open(self.label_list[index]))
          except Exception as e:
              raise ValueError(f"Failed to open mask {self.label_list[index]}: {e}")
          try:
              data = list(self.transforms(*data))
              if data[0] is None or data[1] is None:
                  raise ValueError(f"Transforms returned None for image or mask at {self.image_list[index]}")
          except Exception as e:
              raise ValueError(f"Failed to apply transforms to {self.image_list[index]}: {e}")
          image = data[0]
          label = data[1]
          imn = self.image_list[index].split('/')[-1]
          return (image, label.long(), imt, imn)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_phase = 'img' if self.phase in ['eval', 'test', 'predict'] else self.phase
        self.image_list = get_list_dir(image_phase, 'img', self.data_dir)
        if not self.inference_only and self.phase != 'predict':
            self.label_list = get_list_dir(image_phase, 'mask', self.data_dir)
            assert len(self.image_list) == len(self.label_list), f"Number of images ({len(self.image_list)}) and masks ({len(self.label_list)}) must match in {self.phase}"

def get_list_dir(phase, type_data, data_dir):
    data_dir = osp.join(data_dir, type_data)
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory {data_dir} does not exist")
    exts = ['.png', '.jpg', '.jpeg']
    data_list = []
    for ext in exts:
        data_list.extend(sorted(glob.glob(osp.join(data_dir, f'*{ext}'))))
    return data_list