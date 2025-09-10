import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def visualizar_segmentacion(ruta_archivo, corte='axial', indice_corte=None):
    # Cargar la imagen NIfTI
    img = nib.load(ruta_archivo)
    datos = img.get_fdata()
    
    # Mostrar información básica
    print(f"Dimensiones de la imagen: {datos.shape}")
    
    # Seleccionar el corte a visualizar
    if corte == 'axial':
        # Corte en el eje z
        eje = 2
    elif corte == 'coronal':
        # Corte en el eje y
        eje = 1
    elif corte == 'sagital':
        # Corte en el eje x
        eje = 0
    else:
        raise ValueError("Corte debe ser 'axial', 'coronal' o 'sagital'")
    
    # Seleccionar índice de corte por defecto (medio del volumen)
    if indice_corte is None:
        indice_corte = datos.shape[eje] // 2
    
    # Extraer el corte
    if eje == 0:
        imagen_corte = datos[indice_corte, :, :]
    elif eje == 1:
        imagen_corte = datos[:, indice_corte, :]
    else:
        imagen_corte = datos[:, :, indice_corte]
    
    # Visualizar con matplotlib
    plt.imshow(np.rot90(imagen_corte), cmap='gray')
    plt.title(f'Corte {corte} - índice {indice_corte}')
    plt.axis('off')
    plt.show()

# Ejemplo de uso
ruta = '/IronWolf/oct_seg/oct_nnUNet/predictions/Dataset001_JH/ensemble_postprocessed/JH_001.nii.gz'
visualizar_segmentacion(ruta, corte='axial')
