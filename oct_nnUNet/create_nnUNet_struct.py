import os
import argparse
import subprocess
import sys
from typing import Dict
from batchgenerators.utilities.file_and_folder_operations import save_json, join


def crear_estructura_nnUNet(root_dir: str):
    """
    Crea la estructura de carpetas base para nnUNet dentro del directorio raíz especificado.
    """
    subdirs = ["nnUNet_preprocessed", "nnUNet_raw", "nnUNet_results"]
    for sub in subdirs:
        path = os.path.join(root_dir, sub)
        os.makedirs(path, exist_ok=True)
        print(f"[OK] Carpeta creada: {path}")


def ejecutar_conversion(data_dir: str, dataset_path: str):
    """
    Ejecuta un script de conversión para transformar un dataset en el formato requerido por nnUNet.
    """
    script_path = os.path.join("utils", "dataset_to_nnUNet_format.py")
    if not os.path.isfile(script_path):
        print(f"[ERROR] Script no encontrado: {script_path}")
        sys.exit(1)

    print(f"[INFO] Ejecutando conversión: {script_path} {data_dir} {dataset_path}")
    try:
        subprocess.run(["python", script_path, "--input", data_dir, "--output", dataset_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Falló la conversión: {e}")
        sys.exit(1)


def contar_casos_entrenamiento(imagesTr_dir: str) -> int:
    """
    Cuenta el número de volúmenes en la carpeta imagesTr (entrenamiento).
    """
    if not os.path.isdir(imagesTr_dir):
        print(f"[ERROR] No existe la carpeta: {imagesTr_dir}")
        return 0
    return len([f for f in os.listdir(imagesTr_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])


def crear_dataset_json(output_folder: str, name_dataset: str):
    """
    Crea el archivo dataset.json con los metadatos requeridos por nnUNet.
    """
    from typing import Union, List, Tuple

    def generate_dataset_json(output_folder: str,
                              channel_names: Dict,
                              labels: Dict,
                              num_training_cases: int,
                              file_ending: str,
                              citation: Union[List[str], str] = None,
                              regions_class_order: Tuple[int, ...] = None,
                              dataset_name: str = None,
                              reference: str = None,
                              release: str = None,
                              description: str = None,
                              overwrite_image_reader_writer: str = None,
                              license: str = 'No license specified',
                              converted_by: str = "Unknown",
                              **kwargs):

        # Convertir claves a string
        channel_names = {str(k): v for k, v in channel_names.items()}

        # Asegurar que los valores de labels sean enteros
        for key, val in labels.items():
            if isinstance(val, (tuple, list)):
                labels[key] = tuple(int(x) for x in val)
            else:
                labels[key] = int(val)

        dataset_json = {
            'channel_names': channel_names,
            'labels': labels,
            'numTraining': num_training_cases,
            'file_ending': file_ending,
            'licence': license,
            'converted_by': converted_by,
            'name': dataset_name,
            'description': description
        }

        if reference:
            dataset_json['reference'] = reference
        if release:
            dataset_json['release'] = release
        if citation:
            dataset_json['citation'] = citation
        if overwrite_image_reader_writer:
            dataset_json['overwrite_image_reader_writer'] = overwrite_image_reader_writer
        if regions_class_order:
            dataset_json['regions_class_order'] = regions_class_order

        save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
        print(f"[OK] dataset.json creado en: {output_folder}")

    imagesTr_dir = os.path.join(output_folder, "imagesTr")
    num_training = contar_casos_entrenamiento(imagesTr_dir)

    generate_dataset_json(
        output_folder=output_folder,
        channel_names={0: "OCT"},
        labels={
            "background": 0,
            "layer1": 1,
            "layer2": 2,
            "layer3": 3,
            "layer4": 4,
            "layer5": 5,
            "layer6": 6,
            "layer7": 7,
            "layer8": 8,
            "layer9": 9
        },
        num_training_cases=num_training,
        file_ending=".nii.gz",
        dataset_name=name_dataset,
        description="Segmentacion OCT 3D con 10 clases",
        license="CC-BY 4.0",
        converted_by="Jose Miguel Angos Meza"
    )


def main():
    parser = argparse.ArgumentParser(description="Prepara la estructura base para nnUNet")
    parser.add_argument("ROOT_DIR", help="Directorio raíz donde crear la estructura nnUNet")
    parser.add_argument("DATA_DIR", nargs="?", help="(Opcional) Ruta al dataset con carpetas train/val/test")
    parser.add_argument("NAME_DATASET", nargs="?", help="(Opcional) Nombre del dataset 'DatasetXXX_Name', ej. Dataset005_Prostate")

    args = parser.parse_args()
    root_dir = os.path.abspath(args.ROOT_DIR)
    crear_estructura_nnUNet(root_dir)

    if (args.DATA_DIR and not args.NAME_DATASET) or (args.NAME_DATASET and not args.DATA_DIR):
        print("[ERROR] Si se proporciona DATA_DIR o NAME_DATASET, deben proporcionarse ambos.")
        sys.exit(1)

    if args.DATA_DIR and args.NAME_DATASET:
        data_dir = os.path.abspath(args.DATA_DIR)
        for sub in ["train", "val", "test"]:
            if not os.path.isdir(os.path.join(data_dir, sub)):
                print(f"[ERROR] DATA_DIR debe contener una subcarpeta '{sub}'")
                sys.exit(1)

        dataset_path = os.path.join(root_dir, "nnUNet_raw", args.NAME_DATASET)
        os.makedirs(dataset_path, exist_ok=True)
        ejecutar_conversion(data_dir, dataset_path)

        # Crear dataset.json automáticamente
        crear_dataset_json(dataset_path, args.NAME_DATASET)


if __name__ == "__main__":
    main()
