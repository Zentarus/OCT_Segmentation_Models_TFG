# Consideraciones generales

Los scripts incluidos en esta carpeta han sido desarrollados como herramientas ad hoc para tareas concretas durante el desarrollo del trabajo. No han sido optimizados ni generalizados para su uso en otros contextos, y pueden contener dependencias, rutas o supuestos específicos de los experimentos realizados. Se recomienda encarecidamente revisar y adaptar el código antes de reutilizarlo.

---

# Descripción de scripts

---

## check_optic_nerve_violations.py

Analiza y compara las violaciones topológicas detectadas por los modelos MGU-Net y nnU-Net en imágenes de retina. Lee los resultados desde archivos CSV, extrae información sobre el tipo de protocolo y el número de slice, y genera gráficos de barras que muestran la media de violaciones por slice para los protocolos FastMac y PPole.

---

## delete_coroid_class.py

Procesa máscaras de segmentación en formato PNG para eliminar la clase correspondiente a la coroides (valor 255), reasignándola al fondo (0). Posteriormente remapea los valores de intensidad a índices de clase (0-8) y guarda las imágenes procesadas en un subdirectorio.

---

## extract_nifti.py

Convierte volúmenes en formato NIfTI (.nii.gz) a imágenes individuales en formato PNG (B-scans). Normaliza las intensidades y guarda cada slice en una carpeta específica para cada caso.

---

## graph_images_compare.py

Permite comparar visualmente imágenes de segmentación de diferentes casos y modelos. Carga imágenes, aplica un mapa de colores personalizado para las distintas clases y genera una figura con varias filas y columnas para facilitar la comparación visual entre casos seleccionados.

---

## graph_topography_metric.py

Genera gráficos comparativos de métricas topológicas (violaciones totales, columnas erróneas, porcentaje de error) para los modelos MGU-Net y nnU-Net, diferenciando entre protocolos FastMac y PPole. Lee los datos desde archivos CSV y produce figuras con y sin outliers.

---

## graph_val_test.py

Visualiza y compara métricas de segmentación (por ejemplo, DICE) entre los conjuntos de validación y test, y entre modelos. 

---

## interesting_metrics.py

Analiza métricas topológicas de los modelos MGU-Net y nnU-Net, identificando valores mínimos no nulos y realizando un análisis descriptivo de los resultados. Lee los datos desde archivos CSV y muestra información relevante por consola.

---

## make_special_volumes.py

Agrupa imágenes PNG de cada paciente y crea volúmenes NIfTI (.nii.gz) por paciente, organizando las imágenes por tamaño y asegurando el orden correcto. Utilizado para estructurar el conjunto de datos de imagenes especiales y peripapilar para la entrada de nnU-Net.

---

## plot_histogram_violations.py

Compara la distribución de violaciones topológicas por columna entre dos imágenes de segmentación (por ejemplo, nnU-Net vs MGU-Net). Utiliza funciones auxiliares del script `topology_violation.py` y genera histogramas para cada imagen.

---

## resample_mgunet.py

Redimensiona imágenes de segmentación a un tamaño objetivo (por defecto 1024x512 píxeles) y guarda las imágenes reescaladas en un directorio de salida. 

---

## topology_violation.py

Define la clase `TopologyMetric`, que implementa el cálculo de métricas topológicas sobre segmentaciones de retina. Permite obtener posiciones de capas, detectar violaciones topológicas y calcular métricas asociadas a partir de máscaras de segmentación. Incluye utilidades para normalizar y procesar las máscaras.
