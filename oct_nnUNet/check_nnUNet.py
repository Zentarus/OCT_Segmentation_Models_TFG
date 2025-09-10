# -*- coding: utf-8 -*-
import os
import argparse
import re
import sys
import subprocess

def validar_codigo_dataset(value):
    """Valida que el codigo del dataset sea un numero de 3 digitos."""
    if not re.fullmatch(r"\d{3}", value):
        raise argparse.ArgumentTypeError("El codigo del dataset debe tener exactamente 3 digitos numericos (ej. 001, 042, 123).")
    return value

def main():
    parser = argparse.ArgumentParser(description="Configura las variables de entorno para nnUNet y ejecuta verificacion.")

    parser.add_argument(
        "--base-dir", "-b",
        required=True,
        help="Ruta base que contiene las carpetas 'nnUNet_raw', 'nnUNet_preprocessed' y 'nnUNet_results'."
    )
    parser.add_argument(
        "--dataset-code", "-d",
        required=True,
        type=validar_codigo_dataset,
        help="Codigo numerico de 3 digitos asociado al dataset (ej. 005)."
    )

    args = parser.parse_args()
    base_dir = os.path.abspath(args.base_dir)
    dataset_code = args.dataset_code

    required_dirs = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    for subdir in required_dirs:
        path = os.path.join(base_dir, subdir)
        if not os.path.isdir(path):
            print(f"[ERROR] Falta la carpeta requerida: {path}")
            sys.exit(1)

    script_env = os.path.join(os.path.dirname(__file__), "utils", "set_env_nnUNet.sh")

    if not os.path.isfile(script_env):
        print(f"[ERROR] No se encontro el script de entorno: {script_env}")
        sys.exit(1)

    print("[INFO] Ejecutando verificacion del dataset con entorno cargado...")

    # Comando bash para establecer variables de entorno y ejecutar verificacion
    bash_command = f'''
    source "{script_env}" "{base_dir}" && \
    export nnUNet_dataset={dataset_code} && \
    nnUNetv2_plan_and_preprocess -d {dataset_code} --verify_dataset_integrity
    '''

    try:
        subprocess.run(["bash", "-i", "-c", bash_command], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Fallo la ejecucion de nnUNetv2_plan_and_preprocess: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
