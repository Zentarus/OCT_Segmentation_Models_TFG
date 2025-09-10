#!/bin/bash

# Script: set_nnUNet_env.sh
# Uso: source set_nnUNet_env.sh /ruta/a/directorio/base

# === Validar argumento ===
if [ -z "$1" ]; then
    echo "Uso: source set_nnUNet_env.sh /ruta/a/directorio/base"
    return 1 2>/dev/null || exit 1
fi

# Convertir a ruta absoluta
BASE_DIR="$(cd "$1" && pwd)"

# === Definir rutas absolutas ===
export nnUNet_raw="${BASE_DIR}/nnUNet_raw"
export nnUNet_preprocessed="${BASE_DIR}/nnUNet_preprocessed"
export nnUNet_results="${BASE_DIR}/nnUNet_results"

# === Validar existencia de carpetas ===
for var in nnUNet_raw nnUNet_preprocessed nnUNet_results; do
    dir="${!var}"
    if [ ! -d "$dir" ]; then
        echo "[ERROR] No existe la carpeta: $dir"
        return 1 2>/dev/null || exit 1
    fi
done

# === Confirmar ===
echo "[OK] Variables de entorno configuradas:"
echo "  nnUNet_raw           = $nnUNet_raw"
echo "  nnUNet_preprocessed  = $nnUNet_preprocessed"
echo "  nnUNet_results       = $nnUNet_results"
