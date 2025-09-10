# Autor: José Miguel Angós Meza
# Fecha: 08/09/25

"""
Este script redimensiona imágenes de segmentación a un tamaño objetivo (por defecto 1024x512 píxeles) y guarda las imágenes reescaladas en un directorio de salida.
"""

import os
from PIL import Image

input_dir = "imagenes_servet/special"
output_dir = "special_inference/mgunet"
os.makedirs(output_dir, exist_ok=True)

target_size = (1024, 512)

# Procesa cada imagen en el directorio de entrada
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    # Comprueba si el archivo es una imagen
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        with Image.open(input_path) as img:
            # Redimensiona la imagen
            img_resized = img.resize(target_size, Image.LANCZOS)
            # Guarda la imagen redimensionada
            output_path = os.path.join(output_dir, filename)
            img_resized.save(output_path)

print("Redimensionamiento completo. Las imagenes redimensionadas se guardaron en:", output_dir)