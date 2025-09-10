# Autor: José Miguel Angós Meza
# Fecha: 08/09/25

"""
Este script compara las violaciones topológicas detectadas por los modelos MGU-Net y nnU-Net en imágenes de retina.
Genera gráficos de barras con la media de violaciones por slice para los protocolos FastMac y PPole.
"""

import pandas as pd
import matplotlib.pyplot as plt

def extract_type(imagen):
    """
    Extrae el tipo de protocolo (FastMac o PPole) del nombre de la imagen.
    Parámetros:
        imagen (str): Nombre del archivo de imagen.
    Devueve:
        str: Tipo de protocolo.
    """
    parts = imagen.split('_')
    return parts[parts.index('slice') - 1]

def extract_slice(imagen):
    """
    Extrae el número de slice del nombre de la imagen.
    Parámetros:
        imagen (str): Nombre del archivo de imagen.
    Devueve:
        int: Número de slice.
    """
    return int(imagen.split('_')[-1].split('.png')[0])

df_mgunet = pd.read_csv('topology_results_mgunet.csv')
df_nnunet = pd.read_csv('topology_results_nnunet.csv')

df_mgunet['slice_num'] = df_mgunet['imagen'].apply(extract_slice)
df_mgunet['type'] = df_mgunet['imagen'].apply(extract_type)
df_nnunet['slice_num'] = df_nnunet['imagen'].apply(extract_slice)
df_nnunet['type'] = df_nnunet['imagen'].apply(extract_type)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=False)

# gráfico para FastMac (25 slices)
for df, model, color in [(df_mgunet, 'mgunet', 'blue'), (df_nnunet, 'nnunet', 'orange')]:
    df_fastmac = df[df['type'] == 'FastMac']
    if not df_fastmac.empty:
        means = df_fastmac.groupby('slice_num')['violaciones_totales'].mean()
        means = means.reindex(range(25), fill_value=0)  # asegura todos los slices 0-24
        ax1.bar(means.index, means.values, alpha=0.5, label=model, color=color)

ax1.set_title('Violaciones Promedio por Slice - FastMac')
ax1.set_xlabel('Número de Slice')
ax1.set_ylabel('Violaciones Totales Promedio')
ax1.set_xlim(-0.5, 24.5)
ax1.legend()

# gráfico para PPole (61 slices)
for df, model, color in [(df_mgunet, 'mgunet', 'blue'), (df_nnunet, 'nnunet', 'orange')]:
    df_ppole = df[df['type'] == 'PPole']
    if not df_ppole.empty:
        means = df_ppole.groupby('slice_num')['violaciones_totales'].mean()
        means = means.reindex(range(61), fill_value=0)  # asegura todos los slices 0-60
        ax2.bar(means.index, means.values, alpha=0.5, label=model, color=color)

ax2.set_title('Violaciones Promedio por Slice - PPole')
ax2.set_xlabel('Número de Slice')
ax2.set_ylabel('Violaciones Totales Promedio')
ax2.set_xlim(-0.5, 60.5)
ax2.legend()

plt.tight_layout()
plt.show()