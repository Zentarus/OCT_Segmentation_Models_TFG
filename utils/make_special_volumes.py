# Autor: José Miguel Angós Meza
# Fecha: 08/09/25

"""
Este script agrupa imágenes PNG de cada paciente y crea volúmenes NIfTI (.nii.gz) por paciente, organizando las imágenes por tamaño y asegurando el orden correcto.
"""

import os
import re
import numpy as np
from PIL import Image
import nibabel as nib

SPECIAL_DIR = os.path.join('imagenes_servet', 'special')
OUTPUT_DIR = os.path.join('special_inference', 'nnunet')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ER para extraer el identificador de paciente (todo antes del año)
PATIENT_REGEX = re.compile(r'^([A-Z_]+)_\d{4}_')

patient_files = {}
for fname in os.listdir(SPECIAL_DIR):
	if not fname.lower().endswith('.png'):
		continue
	match = PATIENT_REGEX.match(fname)
	if match:
		patient_id = match.group(1)
		patient_files.setdefault(patient_id, []).append(fname)

# para cada paciente, se crea uno o varios volúmenes NIfTI según el tamaño de las imágenes
for patient_id, files in patient_files.items():
	files_sorted = sorted(files)

	size_groups = {}
	for fname in files_sorted:
		img_path = os.path.join(SPECIAL_DIR, fname)
		img = Image.open(img_path).convert('L')
		size = img.size  # (ancho, alto)
		size_groups.setdefault(size, []).append((fname, np.array(img)))

	for idx, (size, img_list) in enumerate(size_groups.items(), 1):
		# orden por nombre de archivo 
		img_list_sorted = sorted(img_list, key=lambda x: x[0])
		slices = [arr for _, arr in img_list_sorted]
		volume = np.stack(slices, axis=0)
		nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
		out_path = os.path.join(OUTPUT_DIR, f'{patient_id}_vol{idx}.nii.gz')
		nib.save(nifti_img, out_path)
		print(f'Volumen guardado: {out_path} ({len(slices)} slices, tamaño: {size})')
