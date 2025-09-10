# Autor: José Miguel Angós Meza
# Fecha: 08/09/25

"""
Este script permite comparar visualmente imágenes de segmentación de diferentes casos y modelos.
Genera una figura con varias filas y columnas para facilitar la comparación visual entre casos seleccionados.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from PIL import Image
from skimage.transform import resize
import math

base_path = 'imagenes_servet/'
image_names = [
    'GRANADOS_AGUILAR_2019_OD_Control_PPole_slice_049', # Ejemplo de membrana hialoide posterior problemática
    'AMPARO_MONGE_BORDEJE_2016_OD_MS_FastMac_slice_002', # Otro ejemplo de membrana hialoide posterior problemática
    'ANA_ISABEL_BUXEDA_PEREZ_2018_OD_MS_FastMac_slice_024' # Caso con mucho ruido
]

# Clases (0: Fondo, 1-8: clases segmentadas)
classes = ['Fondo', 'RNFL', 'GCL+IPL', 'INL', 'OPL', 'ONL', 'IS', 'OS', 'RPE']

colors = ['black', '#FF3333', '#33CC33', '#3366FF', '#FFFF33', '#CC33FF', '#33FFFF', '#FF9933', '#9933CC']
cmap = ListedColormap(colors)

n_images = len(image_names)
fig, axs = plt.subplots(n_images, 3, figsize=(15, 4 * n_images))

if n_images == 1:
    axs = np.expand_dims(axs, axis=0)

column_titles = ['Original', 'nnU-Net Overlay', 'MGU-Net Overlay']
for j, title in enumerate(column_titles):
    axs[0, j].set_title(title, fontsize=10)

for i, name in enumerate(image_names):
    gt_path = f'{base_path}original/Output_PNG_MGUNET/{name}.png'
    mgu_path = f'{base_path}MGU-Net/processed_images/{name}.png'
    nnu_path = f'{base_path}nnU-Net/processed_images/{name}.png'

    original = np.array(Image.open(gt_path).convert('L'))
    nnu_mask = np.array(Image.open(nnu_path).convert('L'))
    mgu_mask = np.array(Image.open(mgu_path).convert('L'))

    parts = name.split('_')
    protocolo = parts[-3]   
    slice_num = parts[-1].replace('slice', '')  

    # 1. Original
    axs[i, 0].imshow(original, cmap='gray')
    axs[i, 0].text(5, original.shape[0] - 5,
                   f'{i+1}.a) Original - {protocolo} Slice {slice_num}',
                   color='white', fontsize=8, ha='left', va='bottom')
    axs[i, 0].axis('off')

    # 2. nnU-Net overlay
    alpha_nnu = np.where(nnu_mask == 0, 0, 0.75)
    axs[i, 1].imshow(original, cmap='gray', alpha=1.0, zorder=1)
    axs[i, 1].imshow(nnu_mask, cmap=cmap, alpha=alpha_nnu,
                     vmin=0, vmax=8, interpolation='none', zorder=2)
    axs[i, 1].text(5, original.shape[0] - 5,
                   f'{i+1}.b) nnU-Net Overlay',
                   color='white', fontsize=8, ha='left', va='bottom')
    axs[i, 1].axis('off')

    # 3. MGU-Net overlay (resized)
    mgu_mask_resized = resize(mgu_mask, (original.shape[0], original.shape[1]),
                              order=0, preserve_range=True, anti_aliasing=False).astype(int)
    alpha_mgu = np.where(mgu_mask_resized == 0, 0, 0.75)
    axs[i, 2].imshow(original, cmap='gray', alpha=1.0, zorder=1)
    axs[i, 2].imshow(mgu_mask_resized, cmap=cmap, alpha=alpha_mgu,
                     vmin=0, vmax=8, interpolation='none', zorder=2)
    axs[i, 2].text(5, original.shape[0] - 5,
                   f'{i+1}.c) MGU-Net Overlay',
                   color='white', fontsize=8, ha='left', va='bottom')
    axs[i, 2].axis('off')

legend_handles = [mpatches.Patch(color=cmap(k), label=classes[k]) for k in range(1, 9)]

fig.legend(handles=legend_handles, loc='upper center', ncol=4,
           fontsize=8, title='Classes', title_fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('comparative_figure_servet.png', dpi=450)
plt.show()
