import nibabel as nib
import numpy as np
from PIL import Image
import random
import os

nii_file = "/IronWolf/oct_seg/oct_MGUNet/oct_JH/data/Dataset_JH/train/img/JH_003_0000.nii" 

img = nib.load(nii_file)
data = img.get_fdata()  

print("Forma del volumen (shape):", data.shape)  # Ejemplo: (dim0, dim1, dim2), e.g., (512, 1024, 100)

output_dir = "pruebas_b_scans"
os.makedirs(output_dir, exist_ok=True)

# Eje 0: slice = data[random_index, :, :]
dim0 = data.shape[0]
random_idx0 = random.randint(0, dim0 - 1)
slice0 = data[random_idx0, :, :]
slice0_norm = ((slice0 - slice0.min()) / (slice0.max() - slice0.min() + 1e-8) * 255).astype(np.uint8)
Image.fromarray(slice0_norm).save(os.path.join(output_dir, f"slice_eje0_idx{random_idx0}.png"))
print(f"Guardado slice de eje 0 (data[{random_idx0}, :, :]), shape: {slice0.shape}.")

# Eje 1: slice = data[:, random_index, :]
dim1 = data.shape[1]
random_idx1 = random.randint(0, dim1 - 1)
slice1 = data[:, random_idx1, :]
slice1_norm = ((slice1 - slice1.min()) / (slice1.max() - slice1.min() + 1e-8) * 255).astype(np.uint8)
Image.fromarray(slice1_norm).save(os.path.join(output_dir, f"slice_eje1_idx{random_idx1}.png"))
print(f"Guardado slice de eje 1 (data[:, {random_idx1}, :]), shape: {slice1.shape}.")

# Eje 2: slice = data[:, :, random_index]
dim2 = data.shape[2]
random_idx2 = random.randint(0, dim2 - 1)
slice2 = data[:, :, random_idx2]
slice2_norm = ((slice2 - slice2.min()) / (slice2.max() - slice2.min() + 1e-8) * 255).astype(np.uint8)
Image.fromarray(slice2_norm).save(os.path.join(output_dir, f"slice_eje2_idx{random_idx2}.png"))
print(f"Guardado slice de eje 2 (data[:, :, {random_idx2}]), shape: {slice2.shape}.")