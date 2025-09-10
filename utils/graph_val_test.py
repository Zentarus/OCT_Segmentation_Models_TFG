# Autor: José Miguel Angós Meza
# Fecha: 08/09/25

"""
Este script visualiza y compara métricas de segmentación (por ejemplo, DICE) entre los conjuntos de validación y test, y entre modelos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print("Versión de Seaborn:", sns.__version__)

plt.style.use('default')
sns.set_palette("Set2")

df = pd.read_csv('retina_segmentation_data.csv')

val_df = df[df['Set'] == 'Validation']
test_df = df[df['Set'] == 'Test']

classes = val_df['Class'].unique()[1:]

fig, axs = plt.subplots(2, 2, figsize=(14, 12), sharex=False)
bar_width = 0.2
opacity = 0.8

colors = sns.color_palette("Set2", 4)
val_nnU_color = colors[0]  
val_MGU_color = colors[1]  
test_nnU_color = colors[2]  
test_MGU_color = colors[3]  

def plot_metric(ax, df_set, metric, title, sort_by='none'):
    """
    Grafica una métrica para un conjunto específico, permitiendo ordenar las clases.
    Parámetros:
        ax: Eje de matplotlib donde graficar.
        df_set: DataFrame filtrado (validación o test).
        metric: Nombre de la métrica ('Dice', 'PA', etc).
        title: Título del gráfico.
        sort_by: Criterio de ordenación ('none', 'difference', 'highest').
    """
    ordered_classes = classes.copy()
    
    if sort_by == 'difference':
        # por diferencia absoluta entre nnU-Net y MGU-Net
        differences = [(cls, abs(df_set[df_set['Class'] == cls][f'nnU_{metric}'].iloc[0] - df_set[df_set['Class'] == cls][f'MGU_{metric}'].iloc[0])) for cls in classes]
        ordered_classes = [cls for cls, _ in sorted(differences, key=lambda x: x[1], reverse=True)]
    elif sort_by == 'highest':
        # por valores más altos (promedio de nnU-Net y MGU-Net)
        values = [(cls, (df_set[df_set['Class'] == cls][f'nnU_{metric}'].iloc[0] + df_set[df_set['Class'] == cls][f'MGU_{metric}'].iloc[0]) / 2) for cls in classes]
        ordered_classes = [cls for cls, _ in sorted(values, key=lambda x: x[1], reverse=True)]


    x = np.arange(len(ordered_classes))
    nnU_data = [df_set[df_set['Class'] == cls][f'nnU_{metric}'].iloc[0] for cls in ordered_classes]
    MGU_data = [df_set[df_set['Class'] == cls][f'MGU_{metric}'].iloc[0] for cls in ordered_classes]
    
    ax.bar(x - 0.5 * bar_width, nnU_data, bar_width, alpha=opacity, color=val_nnU_color if 'Val' in title else test_nnU_color, label='nnU-Net')
    ax.bar(x + 0.5 * bar_width, MGU_data, bar_width, alpha=opacity, color=val_MGU_color if 'Val' in title else test_MGU_color, label='MGU-Net')
    
    ax.set_ylabel('Porcentaje (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_classes, rotation=45, ha='right')
    ax.set_ylim(80, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize='small')

# Datos para Dice (Validación) - Orden por diferencia (predeterminado)
plot_metric(axs[0, 0], val_df, 'Dice', 'Dice - Validación', sort_by='highest')

# Datos para Dice (Test) - Orden por diferencia
plot_metric(axs[0, 1], test_df, 'Dice', 'Dice - Test', sort_by='highest')

# Datos para PA (Validación) - Orden por diferencia
plot_metric(axs[1, 0], val_df, 'PA', 'PA - Validación', sort_by='highest')

# Datos para PA (Test) - Orden por diferencia
plot_metric(axs[1, 1], test_df, 'PA', 'PA - Test', sort_by='highest')

plt.tight_layout(pad=2.0)
plt.savefig('retina_segmentation_comparison.png', dpi=300)
plt.show()