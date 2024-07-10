import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
from utils import load_images_and_labels
from train import train_and_evaluate_model, evaluate_model_on_test_data

# =============================================================================
# Proyecto desarrollado como parte del Trabajo de Fin de Estudio
# Máster de Inteligencia Artificial de UNIR
# Autor: Javier López Peña
# Fecha: 10 - 07 - 2024
# =============================================================================

# Nota: Este proyecto ha sido desarrollado íntegramente por el alumno,
# figurando como único autor del código y propietario de sus derechos.

# Script principal para entrenar y validar el modelo


if __name__ == "__main__":
    # Paths
    tumor_dir = r'C:\projects\TFM\gestionDataset\imagenesPreprocesadas3\tumor'
    no_tumor_dir = r'C:\projects\TFM\gestionDataset\imagenesPreprocesadas3\no_tumor'
    labels_dir = r'C:\projects\TFM\gestionDataset\imagenesPreprocesadas3\tumor\labels'
    visualization_dir_256 = r'C:\projects\TFM\gestionDataset\imagenesConEtiquetas256\tumor'

    # Cargar las imágenes y etiquetas
    images, labels, image_paths = load_images_and_labels(tumor_dir, no_tumor_dir, labels_dir)

    # Dividir las imágenes con y sin tumor por separado
    images_tumor = [img for img, label in zip(images, labels) if label[0] == 1]
    labels_tumor = [label for label in labels if label[0] == 1]
    image_paths_tumor = [path for path, label in zip(image_paths, labels) if label[0] == 1]

    images_no_tumor = [img for img, label in zip(images, labels) if label[0] == 0]
    labels_no_tumor = [label for label in labels if label[0] == 0]
    image_paths_no_tumor = [path for path, label in zip(image_paths, labels) if label[0] == 0]

    # Dividir el 80% de cada conjunto de datos para entrenamiento y el 20% para test con estratificación
    X_train_tumor, X_test_tumor, y_train_tumor, y_test_tumor, train_paths_tumor, test_paths_tumor = train_test_split(
        images_tumor, labels_tumor, image_paths_tumor, test_size=0.2, random_state=42, stratify=[label[0] for label in labels_tumor]
    )
    X_train_no_tumor, X_test_no_tumor, y_train_no_tumor, y_test_no_tumor, train_paths_no_tumor, test_paths_no_tumor = train_test_split(
        images_no_tumor, labels_no_tumor, image_paths_no_tumor, test_size=0.2, random_state=42, stratify=[label[0] for label in labels_no_tumor]
    )

    # Combinar los conjuntos de entrenamiento y prueba
    X_train = np.array(X_train_tumor + X_train_no_tumor)
    y_train = np.array(y_train_tumor + y_train_no_tumor)
    train_image_paths = train_paths_tumor + train_paths_no_tumor

    X_test = np.array(X_test_tumor + X_test_no_tumor)
    y_test = np.array(y_test_tumor + y_test_no_tumor)
    test_image_paths = test_paths_tumor + test_paths_no_tumor

    # Asegurarse de que los datos estén mezclados
    X_train, y_train, train_image_paths = shuffle(X_train, y_train, train_image_paths, random_state=42)
    X_test, y_test, test_image_paths = shuffle(X_test, y_test, test_image_paths, random_state=42)

    # Configurar K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Entrenar y evaluar el modelo
    model = train_and_evaluate_model(X_train, y_train, kf, train_image_paths, visualization_dir_256)

    # Evaluar el modelo en los datos de test
    evaluate_model_on_test_data(model, X_test, y_test, test_image_paths, visualization_dir_256)
