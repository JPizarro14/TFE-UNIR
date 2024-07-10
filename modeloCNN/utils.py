import os
import numpy as np
import cv2

# =============================================================================
# Proyecto desarrollado como parte del Trabajo de Fin de Estudio
# Máster de Inteligencia Artificial de UNIR
# Autor: Javier López Peña
# Fecha: 10 - 07 - 2024
# =============================================================================

# Nota: Este proyecto ha sido desarrollado íntegramente por el alumno,
# figurando como único autor del código y propietario de sus derechos.

# Script de utilidades para el entenamiento del modelo


def load_images_and_labels(tumor_dir, no_tumor_dir, labels_dir, image_size=64):
    images = []
    labels = []
    image_paths = []

    for img_name in os.listdir(tumor_dir):
        if img_name.endswith('.png') or img_name.endswith('.jpg'):
            img_path = os.path.join(tumor_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (image_size, image_size))
            images.append(img)
            image_paths.append(img_path)
            
            label_path = os.path.join(labels_dir, img_name.replace('.png', '.txt').replace('.jpg', '.txt'))
            if os.path.exists(label_path):
                with open(label_path, 'r') as file:
                    coord = file.readline().strip().split(',')
                    coord = [float(coord[0]), float(coord[1])]
                    labels.append([1, coord])  # 1 indica que hay tumor
            else:
                labels.append([1, [0.0, 0.0]])  # Manejar caso de etiqueta faltante aunque no debería ocurrir

    for img_name in os.listdir(no_tumor_dir):
        if img_name.endswith('.png') or img_name.endswith('.jpg'):
            img_path = os.path.join(no_tumor_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (image_size, image_size))
            images.append(img)
            image_paths.append(img_path)
            labels.append([0, [0.0, 0.0]])  # 0 indica que no hay tumor
    
    return np.array(images), np.array(labels, dtype=object), image_paths

def calculate_euclidean_distance(true_coords, pred_coords):
    return np.sqrt((true_coords[0] - pred_coords[0])**2 + (true_coords[1] - pred_coords[1])**2)

def save_image_with_points(image_path, true_coords, pred_coords, distance, output_folder):
    image = cv2.imread(image_path)
    # No es necesario reescalar, ya que las imágenes de 256x256 se usan directamente
    true_coords = (int(true_coords[0] * 4), int(true_coords[1] * 4))  # Escalar desde 64x64 a 256x256
    pred_coords = (int(pred_coords[0] * 4), int(pred_coords[1] * 4))  # Escalar desde 64x64 a 256x256

    cv2.circle(image, true_coords, 3, (0, 255, 0), -1)  # Punto verde para las coordenadas verdaderas
    cv2.circle(image, pred_coords, 3, (0, 0, 255), -1)  # Punto rojo para las coordenadas predichas

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"{distance:05.2f}px_{filename}")
    cv2.imwrite(output_path, image)
