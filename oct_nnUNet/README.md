# Preparación de Estructura nnU-Net + Conversión de Dataset

Este repositorio contiene una serie de scripts para automatizar la preparación del entorno necesario para entrenar modelos con [nnU-Net v2].

Incluye:

- Creación de la estructura de carpetas esperada por nnU-Net.
- Conversión de datasets en formato `.mat` a volúmenes 3D en formato NIfTI (`.nii.gz`).
- Verificación de la integridad del dataset y configuración automática de variables de entorno.
- Lanzamiento del entrenamiento.

---

## Estructura de Carpetas Generada

Al ejecutar el script `create_nnUNet_struct.py`, se crea automáticamente la siguiente estructura dentro del directorio raíz especificado:

```
ROOT_DIR/
├── nnUNet_preprocessed/
├── nnUNet_raw/
│   └── DatasetXXX_Nombre/
│       ├── imagesTr/
│       ├── labelsTr/
│       ├── imagesVal/
│       └── labelsVal/
├── nnUNet_results/
```

- `nnUNet_raw`: Contiene los datos convertidos en formato NIfTI listos para entrenamiento y validación.
- `nnUNet_preprocessed`: Carpeta usada internamente por nnU-Net durante la planificación y entrenamiento.
- `nnUNet_results`: Donde se almacenan los modelos entrenados y sus resultados.

---

## Uso de Scripts

### 1. Solo crear estructura de carpetas

```bash
python create_nnUNet_struct.py ROOT_DIR
```

Esto solo crea las carpetas `nnUNet_raw`, `nnUNet_preprocessed` y `nnUNet_results`.

---

### 2. Crear estructura y convertir un dataset existente

```bash
python create_nnUNet_struct.py ROOT_DIR DATA_DIR NAME_DATASET
```

- `ROOT_DIR`: Directorio raíz donde se generará la estructura.
- `DATA_DIR`: Ruta al dataset que contiene subcarpetas `train/`, `val/` y `test/` con archivos `.mat`.
- `NAME_DATASET`: Nombre para la carpeta del dataset. Debe seguir la convención: `DatasetXXX_Nombre`. Ejemplo: `Dataset005_Prostate`.

**Ejemplo de dataset de entrada** (`DATA_DIR`):

```
DATA_DIR/
├── train/
│   ├── hc02_spectralis_macula_v1_s1_R_slice_01.mat
│   ├── hc02_spectralis_macula_v1_s1_R_slice_02.mat
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

---

## Proceso de Conversión

El script `utils/dataset_to_nnUNet_format.py`:

- Agrupa slices por paciente.
- Apila los slices en volúmenes 3D.
- Guarda las imágenes en `imagesTr/`, `imagesVal/`, `imagesTs/`.
- Guarda las máscaras (labels) en `labelsTr/`, `labelsVal/`, `labelsTs/`.

Formato de nombres esperado:

```
JH_001_0000.nii.gz   # Imagen
JH_001.nii.gz        # Etiqueta
```

---

## Salida esperada

Tras la conversión, tendremos por ejemplo:

```
ROOT_DIR/
└── nnUNet_raw/
    └── Dataset005_Prostate/
        ├── imagesTr/
        │   ├── JH_001_0000.nii.gz
        │   ├── JH_002_0000.nii.gz
        │   └── ...
        ├── labelsTr/
        │   ├── JH_001.nii.gz
        │   ├── JH_002.nii.gz
        │   └── ...
        ├── imagesVal/
        └── labelsVal/
```

Este formato es el que espera nnU-Net para su entrenamiento.

---

## Verificación del Dataset (`check_nnUNet.py`)

Este script **debe ejecutarse antes del entrenamiento**. Realiza:

- Verificación de la existencia de las carpetas `nnUNet_raw`, `nnUNet_preprocessed` y `nnUNet_results`.
- Llamada al script `set_env_nnUNet.sh` (dentro de `utils/`) para configurar las variables de entorno necesarias.
- Verificación de la integridad del dataset usando `nnUNetv2_plan_and_preprocess`.

```bash
python check_nnUNet.py --base-dir ROOT_DIR --dataset-code 005
```

---

## Entrenamiento (`train_nnUNet.py`)

Este script lanza el entrenamiento usando `nnUNetv2_train`.

```bash
python train_nnUNet.py Dataset005_Prostate 3d_fullres 0
```

Parámetros disponibles:

- `dataset`:    Nombre o ID del dataset (ej. `Dataset005_Prostate`)
- `config`:     Configuración (`2d`, `3d_lowres`, `3d_fullres`)
- `fold`:       Número de fold (`0`–`4`) o `"all"` para todos.
- `--gpus`:                 Número de GPUs (default=1).
- `--pretrained`:           Ruta al checkpoint preentrenado (opcional).
- `--continue_training`:    Si se desea continuar un entrenamiento anterior.

Ejemplo para entrenar en todos los folds:

```bash
python train_nnUNet.py Dataset005_Prostate 3d_fullres all
```

---

## Ejemplo de flujo completo

```bash
python create_nnUNet_struct.py /home/usuario/nnunet_project /home/usuario/mis_datos Dataset001_JHopkins
python check_nnUNet.py --base-dir /home/usuario/nnunet_project --dataset-code 001
python train_nnUNet.py Dataset001_JHopkins 3d_fullres 0
```

---

## Notas adicionales

- El script de conversión espera que cada archivo `.mat` tenga una clave `"slice"` (imagen) y `"layers"` (máscara).
- **MUY IMPORTANTE:** Es necesario exportar correctamente las variables de entorno (ruta a las carpetas `nnUNet_*`). El script `check_nnUNet.py` se encarga de esto mediante `set_env_nnUNet.sh`, aunque recomiendo ejecutar el programa por separado por si acaso.
- Para conocer los detalles de estas implementaciones, recomiendo leer la documentación que ofrece nnUNetv2 en la carpeta "documentation", donde se detalla en profundidad los aspectos, directrices 
  y pasos a realizar para estructurar y poner a punto toda la arquitectura.
- Para realizar inferencia sobre otros conjuntos recomiendo realizarlo con los comandos propios que ofrece nnUNetv2 y leer la documentación asociada.

