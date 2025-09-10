# -*- coding: utf-8 -*-
import os
import numpy as np
import sys

# Forzar UTF-8 en salida estándar (solo para Python 3.7+)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Ruta base de resultados
BASE_PATH = "/IronWolf/oct_seg/oct_nnUNet/oct_JH/nnUNet_results/Dataset001_JH"

# Configuraciones a revisar
configs = [
    "nnUNetTrainer_50epochs__nnUNetPlans__2d",
    "nnUNetTrainer_50epochs__nnUNetPlans__3d_lowres",
    "nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres",
    "nnUNetTrainer_50epochs__nnUNetPlans__3d_cascade_fullres"
]

# Lista para guardar archivos dañados
corrupt_files = []

print("Iniciando verificacion de archivos .npz...")
print()

# Recorre cada configuración y cada fold
for config in configs:
    for fold in range(5):
        val_dir = os.path.join(BASE_PATH, config, f"fold_{fold}", "validation")
        if not os.path.isdir(val_dir):
            print(f"Advertencia: no se encontro la carpeta {val_dir}")
            continue

        for fname in os.listdir(val_dir):
            if fname.endswith(".npz"):
                fpath = os.path.join(val_dir, fname)
                try:
                    with np.load(fpath) as npz:
                        _ = npz['probabilities']
                except Exception as e:
                    print(f"Archivo corrupto: {fpath} - Error: {str(e)}")
                    corrupt_files.append(fpath)

# Resultado final
print("\nVerificacion finalizada.")
if corrupt_files:
    print(f"Se encontraron {len(corrupt_files)} archivo(s) .npz corrupto(s).")
else:
    print("No se encontraron archivos .npz corruptos.")
