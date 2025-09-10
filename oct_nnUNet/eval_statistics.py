import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mapeo de etiquetas numéricas a nombres de capas anatómicos
LAYER_MAP = {
    "0": "Vitreous",      # Humor vítreo
    "1": "RNFL",          # Retina nerve fiber layer
    "2": "GCL + IPL",     # Ganglion cell layer + Inner plexiform layer
    "3": "INL",           # Inner nuclear layer
    "4": "OPL",           # Outer plexiform layer
    "5": "ONL",           # Outer nuclear layer
    "6": "IS",            # Inner photoreceptor segments
    "7": "OS",            # Outer photoreceptor segments
    "8": "RPE",           # Retinal pigment epithelium
    "9": "Sclera"         # Esclera
}


def parse_args():
    """
    Definición de argumentos del programa.
    Requiere un archivo JSON y un directorio de salida.
    """
    parser = argparse.ArgumentParser(description="Procesar métricas de segmentación y generar resultados.")
    parser.add_argument('json_file', type=Path, help="Archivo JSON con las métricas.")
    parser.add_argument('output_dir', type=Path, help="Directorio de salida para guardar métricas y gráficos.")
    parser.add_argument('--include-extremes', action='store_true',
                        help="Incluir capas 0 (Vitreous) y 9 (Sclera) en el análisis.")
    return parser.parse_args()


def compute_precision_sensitivity(row):
    """
    Calcula Precision y Sensitivity a partir de TP, FP y FN.
    """
    tp, fp, fn = row['TP'], row['FP'], row['FN']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, sensitivity


def load_and_process(json_file, include_extremes):
    """
    Carga los datos del archivo JSON y genera DataFrames con las métricas por capa y por caso.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    summary_data = []
    for key, val in data.get('mean', {}).items():
        if not include_extremes and key in ["0", "9"]:
            continue
        layer_name = LAYER_MAP.get(key, f"Layer {key}")
        precision, sensitivity = compute_precision_sensitivity(val)
        summary_data.append({
            'layer': layer_name,
            'Dice': val['Dice'],
            'Precision': precision,
            'Sensitivity': sensitivity,
            'TP': val['TP'],
            'FP': val['FP'],
            'FN': val['FN']
        })
    df_summary = pd.DataFrame(summary_data)

    cases = []
    for case in data.get('metric_per_case', []):
        metrics = case['metrics']
        for key, val in metrics.items():
            if not include_extremes and key in ["0", "9"]:
                continue
            layer_name = LAYER_MAP.get(key, f"Layer {key}")
            precision, sensitivity = compute_precision_sensitivity(val)
            cases.append({
                'case': Path(case['prediction_file']).stem,
                'layer': layer_name,
                'Dice': val['Dice'],
                'Precision': precision,
                'Sensitivity': sensitivity,
                'TP': val['TP'],
                'FP': val['FP'],
                'FN': val['FN']
            })
    df_cases = pd.DataFrame(cases)

    fg_data = None
    if 'foreground_mean' in data:
        fg = data['foreground_mean']
        precision, sensitivity = compute_precision_sensitivity(fg)
        fg_data = {
            'Dice': fg['Dice'],
            'Precision': precision,
            'Sensitivity': sensitivity,
            'TP': fg['TP'],
            'FP': fg['FP'],
            'FN': fg['FN']
        }

    return df_summary, df_cases, fg_data


def export_to_csv(df, filepath):
    """
    Guarda un DataFrame como archivo CSV.
    """
    df.to_csv(filepath, index=False)
    print(f"[INFO] Guardado CSV: {filepath}")


def plot_metrics_per_layer(df_cases, output_dir, include_extremes=False):
    """
    Genera boxplots y barplots mejorados para cada métrica por capa.
    Excluye por defecto las capas 0 (Vitreous) y 9 (Sclera), a menos que se indique lo contrario.
    Añade valores numéricos a las barras, leyenda y estilo adecuado para presentación.
    """
    if not include_extremes:
        df_cases = df_cases[~df_cases['layer'].isin(['Vitreous', 'Sclera'])]

    metrics = ['Dice', 'Precision', 'Sensitivity']
    sns.set(style="whitegrid", font_scale=1.2)

    for metric in metrics:
        # Boxplot
        plt.figure(figsize=(12, 7))
        ax = sns.boxplot(
            x='layer',
            y=metric,
            data=df_cases,
            palette='viridis'
        )
        ax.set_title(f'Distribución de {metric} por Capa', fontsize=16)
        ax.set_xlabel('Capa', fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        file = output_dir / f'{metric.lower()}_boxplot.png'
        plt.savefig(file, dpi=300)
        print(f"[INFO] Guardado gráfico: {file}")
        plt.close()

        # Barplot con anotaciones
        plt.figure(figsize=(12, 7))
        mean_data = df_cases.groupby('layer')[metric].agg(['mean', 'std']).reset_index()
        mean_data = mean_data.sort_values(by='mean', ascending=False)

        ax = sns.barplot(
            x='layer',
            y='mean',
            data=mean_data,
            palette='mako',
            errorbar='sd'
        )

        for i, row in mean_data.iterrows():
            value = f"{row['mean']:.3f} ± {row['std']:.3f}"
            ax.text(
                i,
                row['mean'] + 0.01,
                value,
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        ax.set_title(f'{metric} medio ± desviación estándar por capa', fontsize=16)
        ax.set_xlabel('Capa', fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        file = output_dir / f'{metric.lower()}_barplot.png'
        plt.savefig(file, dpi=300)
        print(f"[INFO] Guardado gráfico: {file}")
        plt.close()


def main():
    """
    Ejecuta todo el flujo: carga datos, exporta métricas, genera gráficos.
    """
    args = parse_args()

    metrics_dir = args.output_dir / "metrics"
    plots_dir = metrics_dir / "plots_imgs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_summary, df_cases, fg_data = load_and_process(args.json_file, args.include_extremes)

    export_to_csv(df_summary, metrics_dir / 'summary_per_layer.csv')
    export_to_csv(df_cases, metrics_dir / 'metrics_per_case.csv')

    if fg_data:
        df_fg = pd.DataFrame([fg_data])
        export_to_csv(df_fg, metrics_dir / 'foreground_mean.csv')

    plot_metrics_per_layer(df_cases, plots_dir, include_extremes=args.include_extremes)


if __name__ == '__main__':
    main()
