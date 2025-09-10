# OCT_Segmentation_Models_TFG
---
## oct_MGUNet
Contiene el código, utilidades y datos asociados al modelo MGU-Net. Incluye scripts de generación de información, carpetas de datos y resultados de inferencia, así como utilidades específicas para el preprocesado y análisis de datos.

- `oct_JH/`: Subcarpetas para checkpoints, datos y resultados de inferencia.
- `SERVET_MGUNET/`: Imágenes adaptadas a la red sobre el dataset SERVET.
- `utils/`: Herramientas y scripts auxiliares para el manejo y análisis de datos.

---

## oct_nnUNet
Incluye todo lo necesario para el entrenamiento, evaluación y análisis del modelo nnU-Net. Contiene scripts de entrenamiento, evaluación, visualización y métricas, así como la estructura de datos y resultados generados.

**IMPORTANTE:**  
> Los datos del **dataset de John Hopkins** (contenido de directorio "nnUNet_raw") **NO están incluidos en este repositorio** debido a su gran tamaño.
> Deben descargarse manualmente desde el siguiente enlace de Google Drive:  
>  https://drive.google.com/drive/folders/1rqZhwFiDU08k7adqdkeuSxVNZD7aI1m_?usp=sharing 

- `oct_JH/`: Estructura de datos preprocesados, datos brutos y resultados de nnU-Net.
- `SERVET_NNUNET/`: Volúmenes NIfTI con datos del dataset SERVET.
- `utils/`: Scripts de apoyo para el procesamiento y análisis de resultados.

---

## RNNs
Carpeta destinada a la experimentación y desarrollo de modelos de redes neuronales recurrentes (RNN). Incluye implementaciones y utilidades tanto para MGU-Net como para nnU-Net en subcarpetas independientes.

- `MGU-Net/`: Código y recursos específicos para el modelo MGU-Net.
- `nnUNet/`: Código y recursos específicos para el modelo nnU-Net.

---

## utils
Contiene scripts y utilidades generales desarrollados para el preprocesado, análisis y visualización de datos y resultados. Incluye herramientas ad hoc para tareas concretas, como eliminación de clases, extracción de imágenes, análisis topológico, generación de gráficos y otras funciones auxiliares.

---
