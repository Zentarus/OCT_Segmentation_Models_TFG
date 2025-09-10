import argparse
import subprocess
import sys
import os

# Importar funcion para verificar la GPU desde el modulo utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from verify_gpu import verificar_gpu  

def lanzar_entrenamiento(dataset, config, fold, num_gpus=1, pretrained=None, continuar=False):
    """
    Ejecuta el comando de entrenamiento de nnUNetv2 con los parametros indicados.

    Parametros:
    - dataset (str): Nombre o ID del dataset (ej. 'Dataset005_Prostate')
    - config (str): Configuracion del modelo (ej. '2d', '3d_lowres', '3d_fullres')
    - fold (int|str): Fold especifico a entrenar (0-4) o "all"
    - num_gpus (int): Numero de GPUs a utilizar (default=1)
    - pretrained (str): Ruta al checkpoint preentrenado (opcional)
    - continuar (bool): Si es True, continua el entrenamiento desde un checkpoint existente
    """
    comando = [
        "nnUNetv2_train",
        dataset,
        config,
        str(fold),
        "-num_gpus", str(num_gpus)
    ]

    if pretrained:
        comando += ["-pretrained_weights", pretrained]

    if continuar:
        comando.append("--c")

    print("\n[INFO] Ejecutando comando:")
    print(" ".join(comando))

    try:
        subprocess.run(comando, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error durante el entrenamiento: {e}")
        sys.exit(1)

def main():
    """
    Funcion principal: procesa los argumentos de entrada, verifica GPU,
    configura el entorno, y lanza el entrenamiento del modelo.
    """
    parser = argparse.ArgumentParser(description="Lanzador de entrenamiento para nnUNetv2")

    # Argumentos obligatorios
    parser.add_argument("dataset", help="Nombre o ID del dataset (ej: Dataset005_Prostate)")
    parser.add_argument("config", help="Configuracion (ej: 2d, 3d_lowres, 3d_fullres)")
    parser.add_argument("fold", help="Fold (0-4 o 'all')")

    # Argumentos opcionales
    parser.add_argument("-g", "--gpus", type=int, default=1, help="Numero de GPUs a usar (default=1)")
    parser.add_argument("-p", "--pretrained", help="Ruta al checkpoint preentrenado (opcional)")
    parser.add_argument("-c", "--continue_training", action="store_true", help="Continuar entrenamiento anterior")

    args = parser.parse_args()

    # Verificar disponibilidad de GPU
    if not verificar_gpu():
        print("[ERROR] No se detecto GPU. Asegurate de estar en un entorno con CUDA.")
        sys.exit(1)


    # Si se indica "all", se entrena en los 5 folds
    if args.fold.lower() == "all":
        for fold in range(5):
            print(f"\n[INFO] Entrenando fold {fold}")
            lanzar_entrenamiento(
                dataset=args.dataset,
                config=args.config,
                fold=fold,
                num_gpus=args.gpus,
                pretrained=args.pretrained,
                continuar=args.continue_training
            )
    else:
        # Entrenamiento individual de un solo fold
        lanzar_entrenamiento(
            dataset=args.dataset,
            config=args.config,
            fold=args.fold,
            num_gpus=args.gpus,
            pretrained=args.pretrained,
            continuar=args.continue_training
        )

if __name__ == "__main__":
    main()
