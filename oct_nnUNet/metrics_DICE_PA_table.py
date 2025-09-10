import json
import numpy as np
import sys

def calculate_metrics(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    metric_per_case = data.get('metric_per_case', [])
    num_cases = len(metric_per_case)
    classes = [str(i) for i in range(1, 10)]  # Clases "1" a "9"
    
    # Inicializar listas para Dice y PA por clase
    dice_values = {cls: [] for cls in classes}
    pa_values = {cls: [] for cls in classes}
    
    # Extraer valores de cada caso
    for case in metric_per_case:
        metrics = case.get('metrics', {})
        for cls in classes:
            if cls in metrics:
                class_metrics = metrics[cls]
                dice = class_metrics.get('Dice', 0)
                tp = class_metrics.get('TP', 0)
                fn = class_metrics.get('FN', 0)
                
                dice_values[cls].append(dice)
                
                # Calcular PA = TP / (TP + FN) si TP + FN > 0
                if tp + fn > 0:
                    pa = tp / (tp + fn)
                else:
                    pa = 0.0
                pa_values[cls].append(pa)
    
    # Calcular media y std para cada clase
    results = {}
    for cls in classes:
        dice_list = dice_values[cls]
        pa_list = pa_values[cls]
        
        if len(dice_list) > 0:
            dice_mean = np.mean(dice_list)
            dice_std = np.std(dice_list)
            pa_mean = np.mean(pa_list)
            pa_std = np.std(pa_list)
            
            results[cls] = {
                'Dice_mean': dice_mean,
                'Dice_std': dice_std,
                'PA_mean': pa_mean,
                'PA_std': pa_std
            }
        else:
            results[cls] = {
                'Dice_mean': 0,
                'Dice_std': 0,
                'PA_mean': 0,
                'PA_std': 0
            }
    
    return results

def print_results(results):
    print("| Clase | Dice (mean +- std)     | PA (mean +- std)       |")
    print("|-------|-----------------------|-----------------------|")
    for cls in sorted(results.keys(), key=int):
        dice_mean = round(results[cls]['Dice_mean'], 4)
        dice_std = round(results[cls]['Dice_std'], 4)
        pa_mean = round(results[cls]['PA_mean'], 4)
        pa_std = round(results[cls]['PA_std'], 4)
        
        print(f"| {cls}     | {dice_mean} +- {dice_std} | {pa_mean} +- {pa_std} |")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python calculate_metrics.py <ruta_al_archivo.json>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    results = calculate_metrics(json_file)
    print_results(results)