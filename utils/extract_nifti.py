# Autor: José Miguel Angós Meza
# Fecha: 08/09/25

"""
Este script convierte volúmenes NIfTI (.nii.gz) en imágenes PNG (B-scans) normalizadas y las guarda en carpetas por caso.
"""

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def nifti_to_png(input_folder, output_folder):
    """
    Convierte todos los archivos .nii.gz de un directorio en imágenes PNG normalizadas y las guarda por caso.
    Parámetros:
        input_folder (str): Carpeta con los archivos .nii.gz
        output_folder (str): Carpeta donde se guardarán los B-scans en PNG
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii.gz"):
            filepath = os.path.join(input_folder, filename)
            nii = nib.load(filepath)
            volume = nii.get_fdata()
            # normaliza intensidades a [0, 255]
            volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
            volume = (volume * 255).astype(np.uint8)
            case_name = filename.replace(".nii.gz", "")
            case_folder = os.path.join(output_folder, case_name)
            os.makedirs(case_folder, exist_ok=True)
            # eje 0 son los B-scans
            for i in range(volume.shape[0]):
                slice_img = volume[i, :, :]
                output_path = os.path.join(case_folder, f"slice_{i:03d}.png")
                plt.imsave(output_path, slice_img, cmap="gray")
            print(f"Guardado {volume.shape[0]} slices de {filename} en {case_folder}")

if __name__ == "__main__":
    input_folder = "special_inference/inferenced_nnunet"  
    output_folder = "special_inference/inferenced_nnunet"
    nifti_to_png(input_folder, output_folder)
