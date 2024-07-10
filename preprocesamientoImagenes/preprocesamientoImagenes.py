import os
import cv2
import numpy as np

# =============================================================================
# Proyecto desarrollado como parte del Trabajo de Fin de Estudio
# Máster de Inteligencia Artificial de UNIR
# Autor: Javier López Peña
# Fecha: 10 - 07 - 2024
# =============================================================================

# Nota: Este proyecto ha sido desarrollado íntegramente por el alumno,
# figurando como único autor del código y propietario de sus derechos.

# En este script se encuentran las funciones para hacer las imágenes cuadradas
# Aplicar CLAHE y redimensionar a 64x64

def make_square(image):
    height, width = image.shape[:2]
    if width > height:
        offset = (width - height) // 2
        image = image[:, offset:offset + height]
    else:
        offset = (height - width) // 2
        padding = ((offset, offset), (0, 0), (0, 0)) if len(image.shape) == 3 else ((offset, offset), (0, 0))
        image = np.pad(image, padding, mode='constant', constant_values=0)
    return image

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 2:
        return clahe.apply(image)
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def process_image(image):
    image = make_square(image)
    image = apply_clahe(image)
    final_image = cv2.resize(image, (64, 64))  # Redimensionar la imagen final a 64x64 píxeles
    return final_image

def process_images_in_folder(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, file)

                image = cv2.imread(input_path)
                if image is not None:
                    processed_image = process_image(image)
                    cv2.imwrite(output_path, processed_image)

if __name__ == "__main__":
    input_folder_tumor = r'C:\projects\TFM\gestionDataset\particionesDatasetYesNo\tumor'
    output_folder_tumor = r'C:\projects\TFM\gestionDataset\imagenesPreprocesadas3\tumor'
    process_images_in_folder(input_folder_tumor, output_folder_tumor)
    
    input_folder_no_tumor = r'C:\projects\TFM\gestionDataset\particionesDatasetYesNo\no_tumor'
    output_folder_no_tumor = r'C:\projects\TFM\gestionDataset\imagenesPreprocesadas3\no_tumor'
    process_images_in_folder(input_folder_no_tumor, output_folder_no_tumor)
