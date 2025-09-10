# Autor: José Miguel Angós Meza
# Fecha: 08/09/25

"""
Este script genera gráficos comparativos de métricas topológicas para los modelos MGU-Net y nnU-Net, diferenciando entre protocolos FastMac y PPole.
Produce figuras con y sin outliers.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


df_mgu = pd.read_csv('topology_results_mgunet.csv')
df_nnu = pd.read_csv('topology_results_nnunet.csv')
df_mgu['Modelo'] = 'MGU-Net'
df_nnu['Modelo'] = 'nnU-Net'

def extract_protocol(imagen):
    """
    Extrae el protocolo (FastMac o PPole) del nombre de la imagen.
    Parámetros:
        imagen (str): Nombre del archivo de imagen.
    Devueve:
        str: Protocolo.
    """
    if 'FastMac' in imagen:
        return 'FastMac'
    elif 'PPole' in imagen:
        return 'PPole'
    return None

df_mgu['Protocolo'] = df_mgu['imagen'].apply(extract_protocol)
df_nnu['Protocolo'] = df_nnu['imagen'].apply(extract_protocol)

df_combined = pd.concat([df_mgu, df_nnu])

metrics = ['violaciones_totales', 'columnas_erroneas', 'porcentaje_error']
metric_labels = ['Violaciones Totales', 'Columnas Erróneas', 'Porcentaje de Error']

colors = ['#00CED1', '#FF4500', '#C71585', '#32CD32']
combinations = ['MGU-Net-FastMac', 'MGU-Net-PPole', 'nnU-Net-FastMac', 'nnU-Net-PPole']

plt.style.use('seaborn-v0_8-whitegrid')

fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False)

for row, show_outliers in enumerate([True, False]):
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        data = [
            df_mgu[df_mgu['Protocolo'] == 'FastMac'][metric].dropna(),
            df_mgu[df_mgu['Protocolo'] == 'PPole'][metric].dropna(),
            df_nnu[df_nnu['Protocolo'] == 'FastMac'][metric].dropna(),
            df_nnu[df_nnu['Protocolo'] == 'PPole'][metric].dropna()
        ]
        
        # Boxplot
        bp = axs[row, i].boxplot(
            data, labels=['', '', '', ''], patch_artist=True, 
            widths=0.22, showfliers=show_outliers
        )
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)
        
        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1.2)
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1.2)
        
        # Valores atípicos
        for flier in bp['fliers']:
            flier.set(marker='o', color='red', alpha=0.5, markersize=5)
        
        if row == 0:
            axs[row, i].set_title(label, fontsize=14, fontweight='bold', pad=10)
        axs[row, i].set_ylabel(label, fontsize=12)
        axs[row, i].tick_params(axis='both', which='major', labelsize=10)
        axs[row, i].grid(True, linestyle='--', alpha=0.5)

axs[0, 0].set_ylabel("Con espúreos", fontsize=12, fontweight="bold")
axs[1, 0].set_ylabel("Sin espúreos", fontsize=12, fontweight="bold")

legend_elements = [
    Patch(facecolor=colors[0], alpha=0.7, label='MGU-Net-FastMac'),
    Patch(facecolor=colors[1], alpha=0.7, label='MGU-Net-PPole'),
    Patch(facecolor=colors[2], alpha=0.7, label='nnU-Net-FastMac'),
    Patch(facecolor=colors[3], alpha=0.7, label='nnU-Net-PPole')
]
fig.legend(handles=legend_elements, loc='upper center', fontsize=12, 
           title='Modelos y Protocolos', title_fontsize=14, ncol=4, 
           frameon=True, edgecolor='black')

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('boxplots_topologia_doble.png', dpi=300, bbox_inches='tight')
plt.show()
