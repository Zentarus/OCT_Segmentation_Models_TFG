
# Autor: José Miguel Angós Meza
# Fecha: 08/09/25

"""
Este script compara la distribución de violaciones topológicas por columna entre dos imágenes de segmentación.
Utiliza funciones auxiliares del script topology_violation.py y genera histogramas para cada imagen.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from topology_violation import load_and_normalize_mask, TopologyMetric


# Configuración: paths de las imágenes a comparar
IMG_PATH_1 = '...'
IMG_PATH_2 = '...'
N_CAPAS = 8

def get_violations_per_column_img(img_path, num_foreground=8):
    """
    Calcula el número de violaciones topológicas por columna en una imagen de segmentación.
    Parámetros:
        img_path (str): Ruta de la imagen.
        num_foreground (int): Número de capas a considerar.
    Devueve:
        np.ndarray: Array con el número de violaciones por columna.
    """
    metric = TopologyMetric(num_capas=num_foreground)
    try:
        seg = load_and_normalize_mask(img_path, num_foreground=num_foreground)
        _, _, _, violations = metric.compute_metric(seg, normalize=True)
        violations_img = violations.squeeze(0).sum(dim=0).cpu().numpy()  # shape: [ancho]
        return violations_img
    except Exception as e:
        print(f"Error procesando {img_path}: {e}")
        return np.array([])

# MAIN: calcula y grafica histogramas
violations_1 = get_violations_per_column_img(IMG_PATH_1, N_CAPAS)
violations_2 = get_violations_per_column_img(IMG_PATH_2, N_CAPAS)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(violations_1, bins=range(int(violations_1.max())+2), color='royalblue', alpha=0.8, rwidth=0.85, label='nnU-Net')
plt.legend()
plt.xlabel('Violaciones topológicas por columna')
plt.ylabel('Frecuencia (nº columnas)')

plt.subplot(1, 2, 2)
plt.hist(violations_2, bins=range(int(violations_2.max())+2), color='darkorange', alpha=0.8, rwidth=0.85, label='MGU-Net')
plt.legend()
plt.xlabel('Violaciones topológicas por columna')
plt.ylabel('Frecuencia (nº columnas)')

plt.tight_layout()
plt.show()
