import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Proyecto desarrollado como parte del Trabajo de Fin de Estudio
# Máster de Inteligencia Artificial de UNIR
# Autor: Javier López Peña
# Fecha: 10 - 07 - 2024
# =============================================================================

# Nota: Este proyecto ha sido desarrollado íntegramente por el alumno,
# figurando como único autor del código y propietario de sus derechos.

# Script que gestiona las distinas gráficas mostradas en la memoria del TFE


def plot_learning_curves(history, fold):
    # Graficar el Recall para clasificación
    if 'Classification_Output_recall_m' in history.history and 'val_Classification_Output_recall_m' in history.history:
        plt.figure()
        plt.plot(history.history['Classification_Output_recall_m'], label='Train Recall')
        plt.plot(history.history['val_Classification_Output_recall_m'], label='Validation Recall')
        plt.title(f'Fold {fold} - Classification Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(f'classification_recall_fold_{fold}.png')

    # Graficar el MAE para regresión
    if 'Regression_Output_mae' in history.history and 'val_Regression_Output_mae' in history.history:
        plt.figure()
        plt.plot(history.history['Regression_Output_mae'], label='Train MAE')
        plt.plot(history.history['val_Regression_Output_mae'], label='Validation MAE')
        plt.title(f'Fold {fold} - Regression MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.savefig(f'regression_mae_fold_{fold}.png')

    # Graficar la pérdida total
    if 'loss' in history.history and 'val_loss' in history.history:
        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Fold {fold} - Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'total_loss_fold_{fold}.png')

def plot_confusion_matrices(confusion_matrices):
    # Extraer VP, VN, FP, FN de cada fold
    vp = [cm[1, 1] for cm in confusion_matrices]
    vn = [cm[0, 0] for cm in confusion_matrices]
    fp = [cm[0, 1] for cm in confusion_matrices]
    fn = [cm[1, 0] for cm in confusion_matrices]

    # Crear la gráfica de barras
    folds = np.arange(1, len(confusion_matrices) + 1)
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar(folds - width, vp, width, label='VP (Verdaderos Positivos)')
    plt.bar(folds, vn, width, label='VN (Verdaderos Negativos)')
    plt.bar(folds + width, fp, width, label='FP (Falsos Positivos)')
    plt.bar(folds + 2*width, fn, width, label='FN (Falsos Negativos)')

    plt.xlabel('Fold')
    plt.ylabel('Cantidad')
    plt.title('Valores de VP, VN, FP, FN por Fold')
    plt.legend()
    plt.savefig('confusion_matrices_summary.png')

def plot_error_histograms(errors):
    errors = np.array(errors)
    x_errors = errors[:, 0]
    y_errors = errors[:, 1]

    # Calcular histogramas
    bins = np.linspace(min(x_errors.min(), y_errors.min()), max(x_errors.max(), y_errors.max()), 31)
    hist_x, _ = np.histogram(x_errors, bins=bins)
    hist_y, _ = np.histogram(y_errors, bins=bins)

    # Crear la gráfica de "stairs"
    plt.figure(figsize=(10, 6))
    plt.stairs(hist_x, bins, label='Error en X', color='blue', fill=False, alpha=0.7, linewidth=2)
    plt.stairs(hist_y, bins, label='Error en Y', color='orange', fill=False, alpha=0.7, linewidth=2)
    plt.xlabel('Error')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de los Errores en la Predicción de las Coordenadas')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_errors_stairs.png')
    plt.show()
