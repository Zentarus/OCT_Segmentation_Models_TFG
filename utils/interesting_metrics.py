# Autor: José Miguel Angós Meza
# Fecha: 08/09/25

"""
Este script analiza métricas topológicas de los modelos MGU-Net y nnU-Net, identificando valores mínimos no nulos y realizando un análisis descriptivo de los resultados.
Lee los datos desde archivos CSV y muestra información relevante por consola.
"""

import pandas as pd
import numpy as np



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

def get_lowest_nonzero(df, column):
    """
    Busca el valor no nulo más bajo de una columna.
    Parámetros:
        df: DataFrame.
        column: Nombre de la columna.
    Devueve:
        (row, valor): Fila y valor mínimo no nulo, o (None, None) si no hay.
    """
    nonzero = df[df[column] > 0][column]
    if nonzero.empty:
        return None, None
    idx = nonzero.idxmin()
    return df.loc[idx], nonzero.min()

print("ANÁLISIS DE MÉTRICAS TOPOLÓGICAS")
print("=" * 50)


for metric, label in zip(metrics, metric_labels):
    print(f"\nMétrica: {label}")
    print("-" * 40)
    
    # Mayor valor (peor rendimiento)
    max_idx = df_combined[metric].idxmax()
    max_row = df_combined.loc[max_idx].iloc[0] if isinstance(df_combined.loc[max_idx], pd.DataFrame) else df_combined.loc[max_idx]
    print(f"Mayor valor:")
    print(f"  Imagen: {max_row['imagen']}")
    print(f"  Modelo: {max_row['Modelo']}")
    print(f"  Protocolo: {max_row['Protocolo']}")
    print(f"  Valor: {max_row[metric]:.2f}")
    print(f"  Porcentaje de Error: {max_row['porcentaje_error']:.2f}%")
    
    # Menor valor no cero (mejor rendimiento, excluyendo 0)
    min_row, min_value = get_lowest_nonzero(df_combined, metric)
    if min_row is not None:
        min_row = min_row.iloc[0] if isinstance(min_row, pd.DataFrame) else min_row
        print(f"\nMenor valor no cero:")
        print(f"  Imagen: {min_row['imagen']}")
        print(f"  Modelo: {min_row['Modelo']}")
        print(f"  Protocolo: {min_row['Protocolo']}")
        print(f"  Valor: {min_value:.2f}")
        print(f"  Porcentaje de Error: {min_row['porcentaje_error']:.2f}%")
    else:
        print(f"\nMenor valor no cero: No hay valores no cero para {label}")
    
    print(f"\nEstadísticas por Modelo y Protocolo:")
    for model in ['MGU-Net', 'nnU-Net']:
        for protocol in ['FastMac', 'PPole']:
            subset = df_combined[(df_combined['Modelo'] == model) & (df_combined['Protocolo'] == protocol)][metric]
            if not subset.empty:
                print(f"  {model} - {protocol}:")
                print(f"    Media: {subset.mean():.2f}")
                print(f"    Mediana: {subset.median():.2f}")
                print(f"    Máximo: {subset.max():.2f}")
                print(f"    Mínimo: {subset.min():.2f}")
                print(f"    Imágenes analizadas: {len(subset)}")
