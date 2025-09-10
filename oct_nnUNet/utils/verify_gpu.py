# utils/verify_gpu.py
import torch

def verificar_gpu():
    disponible = torch.cuda.is_available()
    if disponible:
        nombre = torch.cuda.get_device_name(0)
        print("CUDA disponible:", disponible)
        print("GPU:", nombre)
    else:
        print("CUDA no disponible.")
    return disponible
