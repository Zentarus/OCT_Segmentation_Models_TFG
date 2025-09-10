# Autor: José Miguel Angós Meza
# Fecha: 08/09/25

"""
Este script define la clase TopologyMetric para el cálculo de métricas topológicas sobre segmentaciones de retina.
Incluye utilidades para normalizar y procesar máscaras, y para calcular violaciones topológicas.
"""

import torch
import numpy as np
from PIL import Image
import os
import glob
import csv


class TopologyMetric:
    """
    Clase para calcular métricas topológicas en segmentaciones de retina.
    Permite obtener posiciones de capas, detectar violaciones topológicas y calcular métricas asociadas.
    """
    def __init__(self, num_capas=8):
        self.num_capas = num_capas
        self.pos_mask = {}
        self.relu = torch.nn.functional.relu

    def get_position_mask(self, template):
        """
        Genera una máscara de posiciones para el cálculo de posiciones de capa.
        Parámetros:
            template (np.ndarray): Array de referencia para dimensiones.
        Devueve:
            torch.Tensor: Máscara de posiciones.
        """
        w = template.shape[-1]  # ancho
        if w in self.pos_mask:
            return self.pos_mask[w]
        column = np.arange(template.shape[-2])  # valores y (0 a altura-1)
        column = np.expand_dims(column, 1)
        mask = np.repeat(column, w, axis=1)
        mask = np.expand_dims(mask, 0)
        mask = np.repeat(mask, self.num_capas, axis=0)
        mask = np.expand_dims(mask, 0)
        self.pos_mask[w] = torch.tensor(mask, dtype=torch.float32)
        return self.pos_mask[w]

    def get_layer_positions(self, sm):
        """
        Calcula la posición de cada capa en cada columna.
        Parámetros:
            sm (torch.Tensor): Segmentación normalizada [batch, num_capas, alto, ancho].
        Devueve:
            torch.Tensor: Posiciones de capa [batch, num_capas, ancho].
        """
        mask = self.get_position_mask(sm)
        return torch.sum(sm * mask, dim=2)

    def get_topology_violations(self, layer_positions):
        """
        Calcula las violaciones topológicas entre capas adyacentes.
        Parámetros:
            layer_positions (torch.Tensor): Posiciones de capa.
        Devueve:
            torch.Tensor: Violaciones topológicas.
        """
        violations = layer_positions[:, :-1, :] - layer_positions[:, 1:, :]
        return self.relu(violations)

    def compute_metric(self, seg, normalize=True):
        """
        Calcula métricas topológicas sobre una segmentación.
        Parámetros:
            seg (torch.Tensor): Segmentación [batch, num_capas, alto, ancho].
            normalize (bool): Si normaliza o no la segmentación.
        Devueve:
            total_violations: Total de violaciones topológicas.
            num_bad_columns: Número de columnas con violaciones.
            positions: Posiciones de capa.
            violations: Array de violaciones.
        """
        if normalize:
            sums = torch.sum(seg, dim=2, keepdim=True)
            sums[sums == 0] = 1  # evitar división por 0
            sm = seg / sums
        else:
            sm = seg
        positions = self.get_layer_positions(sm)
        violations = self.get_topology_violations(positions)
        total_violations = violations.sum()
        has_violation = (violations > 0).any(dim=1)
        num_bad_columns = has_violation.sum()
        return total_violations, num_bad_columns, positions, violations

def load_and_normalize_mask(file_path, num_foreground=8):
    """
    Carga imagen .png, normaliza valores a 0-8 si están escalados, 
    y convierte a tensor [1, num_foreground, h, w].
    """
    img = Image.open(file_path).convert('L') 
    img_np = np.array(img)
    
    # Normalización a 0–8
    if img_np.max() > 8:
        scale_factor = img_np.max() / 8.0
        normalized = np.round(img_np / scale_factor).astype(int)
        labels_np = normalized
    else:
        labels_np = img_np
    
    # Convertir a tensor [1, h, w]
    labels = torch.tensor(labels_np, dtype=torch.long).unsqueeze(0)
    
    # One-hot [1, num_foreground, h, w]
    batch, h, w = labels.shape
    seg = torch.zeros(batch, num_foreground, h, w, dtype=torch.float32)
    for c in range(1, num_foreground + 1):
        seg[:, c-1, :, :] = (labels == c).float()
    
    return seg

def process_directory(directory_path, num_foreground=8, output_file='topology_results.csv'):
    """
    Procesa todas las .png en el directorio, calcula métricas topológicas y guarda resultados en CSV.
    """
    metric = TopologyMetric(num_capas=num_foreground)
    
    png_paths = glob.glob(os.path.join(directory_path, '*.png'))
    if not png_paths:
        print(f"No se encontraron archivos .png en {directory_path}")
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        writer.writerow([
            "imagen", 
            "violaciones_totales", 
            "columnas_erroneas", 
            "porcentaje_error", 
            "porcentaje_acierto"
        ])
        
        for path in png_paths:
            try:
                seg = load_and_normalize_mask(path, num_foreground=num_foreground)
                total_viol, num_bad_cols, positions, violations = metric.compute_metric(seg, normalize=True)
                
                ancho = seg.shape[-1]
                porcentaje_error = (num_bad_cols.item() / ancho) * 100 if ancho > 0 else 0
                porcentaje_acierto = 100 - porcentaje_error
                
                print(f"\nProcesando: {os.path.basename(path)}")
                print(f"Violaciones totales: {total_viol.item():.2f}")
                print(f"Columnas con errores: {num_bad_cols.item()} de {ancho} ({porcentaje_error:.2f}%)")
                print(f"Porcentaje de acierto: {porcentaje_acierto:.2f}%")
                
                writer.writerow([
                    os.path.basename(path), 
                    total_viol.item(), 
                    num_bad_cols.item(), 
                    f"{porcentaje_error:.2f}", 
                    f"{porcentaje_acierto:.2f}"
                ])
            
            except Exception as e:
                print(f"Error en {path}: {str(e)}")
                writer.writerow([os.path.basename(path), "ERROR", str(e), "-", "-"])

directory = 'special_inference/inferenced_mgunet/processed_images'
process_directory(directory, num_foreground=8, output_file='topology_results_mgunet_peripapilar.csv')
