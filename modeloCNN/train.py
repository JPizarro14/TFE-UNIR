import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from model import create_model
from utils import calculate_euclidean_distance, save_image_with_points
from graphics import plot_learning_curves, plot_confusion_matrices, plot_error_histograms

# =============================================================================
# Proyecto desarrollado como parte del Trabajo de Fin de Estudio
# Máster de Inteligencia Artificial de UNIR
# Autor: Javier López Peña
# Fecha: 10 - 07 - 2024
# =============================================================================

# Nota: Este proyecto ha sido desarrollado íntegramente por el alumno,
# figurando como único autor del código y propietario de sus derechos.

# Script donde se define el entrenamiento y la evaluación del modelo

def train_and_evaluate_model(X_train, y_train, kf, image_paths, visualization_dir_256):
    fold_no = 1
    all_euclidean_distances = []
    all_errors = []
    confusion_matrices = []

    os.makedirs('fold_images', exist_ok=True)
    
    for train_index, val_index in kf.split(X_train):
        print(f"Training fold {fold_no} ...")
        
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        val_image_paths = [image_paths[i] for i in val_index]

        y_train_class = np.array([label[0] for label in y_train_fold], dtype=np.float32)
        y_train_reg = np.array([label[1] for label in y_train_fold], dtype=np.float32)
        y_val_class = np.array([label[0] for label in y_val_fold], dtype=np.float32)
        y_val_reg = np.array([label[1] for label in y_val_fold], dtype=np.float32)

        model = create_model()

        history = model.fit(X_train_fold, 
                            {'Classification_Output': y_train_class, 'Regression_Output': y_train_reg},
                            validation_data=(X_val_fold, {'Classification_Output': y_val_class, 'Regression_Output': y_val_reg}),
                            epochs=100, batch_size=32, verbose=2)
        
        plot_learning_curves(history, fold_no)
        
        y_pred_class, y_pred_reg = model.predict(X_val_fold)
        y_pred_class = (y_pred_class > 0.5).astype(int)
        y_pred_reg = np.round(y_pred_reg).astype(int)  # Redondear las predicciones a enteros
        
        cm = confusion_matrix(y_val_class, y_pred_class)
        confusion_matrices.append(cm)
        precision = precision_score(y_val_class, y_pred_class) * 100
        recall = recall_score(y_val_class, y_pred_class) * 100
        f1 = f1_score(y_val_class, y_pred_class) * 100
        
        print(f"Fold {fold_no} - Confusion Matrix:\n{cm}")
        print(f"Fold {fold_no} - Precision: {precision:.2f}%")
        print(f"Fold {fold_no} - Recall: {recall:.2f}%")
        print(f"Fold {fold_no} - F1 Score: {f1:.2f}%")
        
        fold_image_folder = os.path.join('fold_images', f'fold_{fold_no}')
        os.makedirs(fold_image_folder, exist_ok=True)
        
        for true_class, true_coords, pred_class, pred_coords, img_path in zip(y_val_class, y_val_reg, y_pred_class, y_pred_reg, val_image_paths):
            if true_class == 1:  # Solo considerar las imágenes que tienen tumor
                distance = calculate_euclidean_distance(true_coords, pred_coords)
                if distance > 4:
                    image_path_256 = os.path.join(visualization_dir_256, os.path.basename(img_path))
                    save_image_with_points(image_path_256, true_coords, pred_coords, distance, fold_image_folder)
        
        euclidean_distances = [calculate_euclidean_distance(true, pred) for true, pred, true_cls in zip(y_val_reg, y_pred_reg, y_val_class) if true_cls == 1]
        all_euclidean_distances.extend(euclidean_distances)
        
        errors = [(true[0] - pred[0], true[1] - pred[1]) for true, pred, true_cls in zip(y_val_reg, y_pred_reg, y_val_class) if true_cls == 1]
        all_errors.extend(errors)
        
        mean_euclidean_distance = np.mean(euclidean_distances)
        print(f"Fold {fold_no} - Mean Euclidean Distance: {mean_euclidean_distance:.2f}")

        fold_no += 1
    
    overall_mean_euclidean_distance = np.mean(all_euclidean_distances)
    print(f"Overall Mean Euclidean Distance: {overall_mean_euclidean_distance:.2f}")

    plot_confusion_matrices(confusion_matrices)
    plot_error_histograms(all_errors)

    return model

def evaluate_model_on_test_data(model, X_test, y_test, test_image_paths, visualization_dir_256):
    y_test_class = np.array([label[0] for label in y_test], dtype=np.float32)
    y_test_reg = np.array([label[1] for label in y_test], dtype=np.float32)

    y_pred_class, y_pred_reg = model.predict(X_test)
    y_pred_class = (y_pred_class > 0.5).astype(int)
    y_pred_reg = np.round(y_pred_reg).astype(int)  # Redondear las predicciones a enteros

    cm = confusion_matrix(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class) * 100
    recall = recall_score(y_test_class, y_pred_class) * 100
    f1 = f1_score(y_test_class, y_pred_class) * 100
    
    print(f"Test Data - Confusion Matrix:\n{cm}")
    print(f"Test Data - Precision: {precision:.2f}%")
    print(f"Test Data - Recall: {recall:.2f}%")
    print(f"Test Data - F1 Score: {f1:.2f}%")
    
    # Crear la carpeta para guardar las imágenes de prueba
    test_image_folder = 'test_images'
    os.makedirs(test_image_folder, exist_ok=True)
    
    # Validación de la posición del centro del tumor con distancia euclidiana
    euclidean_distances = []
    errors = []
    for true_class, true_coords, pred_class, pred_coords, img_path in zip(y_test_class, y_test_reg, y_pred_class, y_pred_reg, test_image_paths):
        if true_class == 1:  # Solo considerar las imágenes que tienen tumor
            distance = calculate_euclidean_distance(true_coords, pred_coords)
            euclidean_distances.append(distance)
            errors.append((true_coords[0] - pred_coords[0], true_coords[1] - pred_coords[1]))
            if distance > 4:
                image_path_256 = os.path.join(visualization_dir_256, os.path.basename(img_path))
                save_image_with_points(image_path_256, true_coords, pred_coords, distance, test_image_folder)
    
    print(f"Test Data - Euclidean Distances: {euclidean_distances}")
    mean_euclidean_distance = np.mean(euclidean_distances)
    print(f"Test Data - Mean Euclidean Distance: {mean_euclidean_distance:.2f}")
    
    plot_error_histograms(errors)
