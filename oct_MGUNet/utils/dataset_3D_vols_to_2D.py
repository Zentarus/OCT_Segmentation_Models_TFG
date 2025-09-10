import os
import shutil
import nibabel as nib
import numpy as np
from PIL import Image

def nii_to_png_slices(input_nii_path, output_dir):
    nii_img = nib.load(input_nii_path)
    data = nii_img.get_fdata()
    n_slices = data.shape[0]

    base_name = os.path.splitext(os.path.basename(input_nii_path))[0]
    if base_name.endswith('.nii'):
        base_name = os.path.splitext(base_name)[0]

    for i in range(n_slices):
        slice_2d = data[i, :, :]

        if input_nii_path.lower().endswith('mask.nii') or 'mask' in input_nii_path.lower():
            slice_2d = slice_2d.astype(np.uint8)
        else:
            slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
            slice_2d = (slice_norm * 255).astype(np.uint8)

        output_filename = f"{base_name}_slice_{i:03d}.png"
        output_path = os.path.join(output_dir, output_filename)

        im = Image.fromarray(slice_2d)
        im.save(output_path)

def convert_dataset_nii_to_2d(dataset_dir, output_dir):
    # copia estructura de carpetas
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(dataset_dir, output_dir, ignore=shutil.ignore_patterns('*.nii', '*.nii.gz'))

    # procesa cada archivo .nii y .nii.gz en dataset_dir para convertir a png en output_dir
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                nii_path = os.path.join(root, f)
                relative_path = os.path.relpath(root, dataset_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                print(f"Convirtiendo {nii_path} a imagenes 2D en {output_subdir}")
                nii_to_png_slices(nii_path, output_subdir)

if __name__ == "__main__":
    dataset_dir = "/IronWolf/oct_seg/oct_MGUNet/oct_JH/data/Dataset_JH/"
    output_dir = "/IronWolf/oct_seg/oct_MGUNet/oct_JH/data/Dataset_JH_2D/"
    convert_dataset_nii_to_2d(dataset_dir, output_dir)
