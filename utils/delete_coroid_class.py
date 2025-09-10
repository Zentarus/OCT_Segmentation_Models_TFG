# Autor: José Miguel Angós Meza
# Fecha: 08/09/25

"""
Este script elimina la clase de coroides (valor 255) de máscaras de segmentación en PNG, reasignando al fondo y remapeando a índices de clase 0-8.
Guarda las imágenes procesadas en un subdirectorio.
"""

import os
import numpy as np
from PIL import Image

def process_images(input_dir):
    """
    Procesa todas las imágenes PNG de un directorio, elimina la clase de coroides (255) y remapea a clases 0-8.
    Parámetros:
        input_dir (str): Directorio con las imágenes a procesar.
    No Devueve nada. Guarda las imágenes procesadas en un subdirectorio.
    """
    output_dir = os.path.join(input_dir, "processed_images")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if not (os.path.isfile(file_path) and filename.endswith(".png")):
            continue
        if os.path.commonpath([output_dir, file_path]) == output_dir:
            continue
        mask = np.array(Image.open(file_path).convert('L'))

        print(f"Procesando: {file_path}")
        print(f"Valores únicos: {np.unique(mask)}")

        # elimina la clase de coroides (255) y la pone a fondo (0)
        mask[mask == 255] = 0

        # valores de intensidad a clases 0-8
        mask = np.clip(np.round(mask / 28), 0, 8).astype(int)

        image_array = mask.astype(np.uint8)
        processed_image = Image.fromarray(image_array)
        output_path = os.path.join(output_dir, filename)
        processed_image.save(output_path)

    print(f"Proceso completado - Las imagenes se guardaron en: {output_dir}")

if __name__ == "__main__":
    input_directory = input("Introduce el directorio donde se encuentran las imágenes: ").strip()
    if os.path.isdir(input_directory):
        process_images(input_directory)
    else:
        print("El directorio proporcionado no es valido.")