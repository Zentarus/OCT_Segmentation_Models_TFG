import os
import re
import numpy as np
import scipy.io
import nibabel as nib
from collections import defaultdict
from tqdm import tqdm
import argparse

# ---------------------------
# PARSEO DE ARGUMENTOS
# ---------------------------

parser = argparse.ArgumentParser(description="Conversor de .mat a NIfTI para nnUNet")
parser.add_argument("--input", "-i", required=True, help="Ruta a la carpeta del dataset (debe contener train/, val/, test/)")
parser.add_argument("--output", "-o", required=True, help="Ruta de salida donde se guardarán las imágenes NIfTI")
parser.add_argument("--nii", action="store_true", help="Si se usa, los archivos se guardarán como .nii en lugar de .nii.gz")
args = parser.parse_args()

# Asignar rutas desde argumentos
DATASET_ROOT = args.input
OUTPUT_ROOT = args.output
usar_nii_gz = not args.nii
ext = ".nii.gz" if usar_nii_gz else ".nii"

# Crear carpetas necesarias para nnUNet
for folder in ["imagesTr", "labelsTr", "imagesTs", "labelsTs", "imagesVal", "labelsVal"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, folder), exist_ok=True)

# ---------------------------
# FUNCIONES
# ---------------------------

def agrupar_por_paciente(archivos, modo='train'):
    """
    Agrupa archivos .mat por paciente, basándose en un patrón de nombre de archivo.
    """
    grupos = defaultdict(list)
    regex = re.compile(r"(hc\d{2}|ms\d{2})_.*slice_(\d+)")

    paciente_ids = sorted(set(
        re.match(regex, os.path.basename(f).replace(".mat", "")).group(1)
        for f in archivos if re.match(regex, os.path.basename(f).replace(".mat", ""))
    ))

    paciente_a_id = {paciente: f"{idx+1:03d}" for idx, paciente in enumerate(paciente_ids)}

    for f in archivos:
        nombre = os.path.basename(f).replace(".mat", "")
        match = regex.match(nombre)
        if match:
            paciente_id_original, slice_idx = match.groups()
            paciente_id_nuevo = paciente_a_id[paciente_id_original]

            base_nombre = f"JH_{paciente_id_nuevo}"
            grupos[base_nombre].append((int(slice_idx), f))

    return grupos


def procesar_paciente(paciente_id, slices, destino_img, incluir_label=True):
    """
    Convierte los slices .mat de un paciente en un volumen 3D y lo guarda en formato NIfTI.
    """
    slices.sort(key=lambda x: x[0])
    slices_img = []
    slices_lbl = []

    for _, archivo in slices:
        mat = scipy.io.loadmat(archivo)
        img = mat["slice"].astype(np.float32)
        slices_img.append(img)

        if incluir_label:
            lbl = mat["layers"].astype(np.uint8)
            slices_lbl.append(lbl)

    # Guardar imagen
    vol_img = np.stack(slices_img, axis=0)
    nifti_img = nib.Nifti1Image(vol_img, affine=np.eye(4))

    # Añadir _0000 solo para imágenes
    img_filename = f"{paciente_id}_0000{ext}"
    img_path = os.path.join(OUTPUT_ROOT, destino_img, img_filename)
    nib.save(nifti_img, img_path)

    # Guardar etiqueta sin _0000
    if incluir_label:
        vol_lbl = np.stack(slices_lbl, axis=0)
        nifti_lbl = nib.Nifti1Image(vol_lbl, affine=np.eye(4))

        if destino_img == "imagesTr":
            destino_lbl = "labelsTr"
        elif destino_img == "imagesTs":
            destino_lbl = "labelsTs"
        elif destino_img == "imagesVal":
            destino_lbl = "labelsVal"
        else:
            raise ValueError(f"Destino desconocido: {destino_img}")

        lbl_filename = f"{paciente_id}{ext}"
        lbl_path = os.path.join(OUTPUT_ROOT, destino_lbl, lbl_filename)
        nib.save(nifti_lbl, lbl_path)

    print(f"[OK] Guardado: {paciente_id} -> {destino_img}")


# ---------------------------
# EJECUCIÓN PRINCIPAL
# ---------------------------

for split in ["train", "val", "test"]:
    folder = os.path.join(DATASET_ROOT, split)
    if not os.path.isdir(folder):
        print(f"[ADVERTENCIA] Carpeta no encontrada: {folder}")
        continue

    archivos = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mat")]
    grupos = agrupar_por_paciente(archivos, modo=split)

    destino_img = {
        "train": "imagesTr",
        "val": "imagesVal",
        "test": "imagesTs"
    }[split]

    for paciente_id, slices in tqdm(grupos.items(), desc=f"Procesando {split}"):
        procesar_paciente(paciente_id, slices, destino_img, incluir_label=True)
