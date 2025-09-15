import albumentations as A
import numpy as np
from PIL import Image
import cv2

class AlbumentationsTransform:
    def __init__(self, augment=True, target_height=512, target_width=1024):
        self.target_height = target_height
        self.target_width = target_width
        if augment:
            self.transform = A.Compose([
                # 1. Rotacion (p=0.2, recortar para mantener tamano original)
                A.Rotate(limit=(-15, 15), crop_border=True, p=0.2),
                # 2. Ruido gaussiano (p=0.15, varianza U(0.0001,0.1))
                A.GaussNoise(var_limit=(0.0001, 0.1), p=0.15),
                # 3. Gaussian blur (p=0.2, kernel U(3,7), sigma U(0.5,1.5))
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 1.5), p=0.2),
                # 4. Brillo (p=0.15, multiplicativo U(0.7,1.3))
                A.RandomBrightnessContrast(brightness_limit=(0.7-1, 1.3-1), contrast_limit=0, p=0.15),
                # 5. Contraste (p=0.15, multiplicativo U(0.65,1.5))
                A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.65-1, 1.5-1), p=0.15),
                # 6. Gamma (p=0.15 normal, p=0.15 invertido)
                A.OneOf([
                    A.RandomGamma(gamma_limit=(70, 150), p=1.0),
                    A.Compose([
                        A.Lambda(image=lambda x, **kwargs: 1 - x, mask=lambda x, **kwargs: x),  # Invertir imagen
                        A.RandomGamma(gamma_limit=(70, 150), p=1.0),
                        A.Lambda(image=lambda x, **kwargs: 1 - x, mask=lambda x, **kwargs: x)  # Revertir
                    ], p=1.0)
                ], p=0.15),
                # 7. Mirroring (p=0.5, solo eje horizontal para OCT)
                A.HorizontalFlip(p=0.5),
                # 8. Redimensionar al tamano esperado por la red (1024x512)
                A.Resize(height=target_height, width=target_width, interpolation=cv2.INTER_LINEAR),
            ], additional_targets={'mask': 'mask'})
        else:
            self.transform = None

    def __call__(self, image, mask):
        if self.transform:
            # Convertir PIL.Image a NumPy
            image_np = np.array(image)
            mask_np = np.array(mask)
            # Asegurar que image_np sea HxWxC
            if image_np.ndim == 2:
                image_np = image_np[..., np.newaxis]  # Convertir HxW a HxWx1
            # Aplicar transformaciones
            transformed = self.transform(image=image_np, mask=mask_np)
            # Convertir de vuelta a PIL.Image
            image = Image.fromarray(transformed['image'].squeeze())  # Quitar canal si es HxWx1
            mask = Image.fromarray(transformed['mask'].astype(np.uint8))  # Asegurar tipo entero para m�scara
        return image, mask