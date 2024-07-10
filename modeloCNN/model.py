import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from metrics import recall_m

# =============================================================================
# Proyecto desarrollado como parte del Trabajo de Fin de Estudio
# Máster de Inteligencia Artificial de UNIR
# Autor: Javier López Peña
# Fecha: 10 - 07 - 2024
# =============================================================================

# Nota: Este proyecto ha sido desarrollado íntegramente por el alumno,
# figurando como único autor del código y propietario de sus derechos.

# Definición de la arquitectura del modelo


def create_model():
    input_layer = Input(shape=(64, 64, 3), name='Input_Layer')

    # Bloque codificador
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='Conv2D_1')(input_layer)
    x = MaxPooling2D((2, 2), name='MaxPooling2D_1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv2D_2')(x)
    x = MaxPooling2D((2, 2), name='MaxPooling2D_2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='Conv2D_3')(x)
    x = MaxPooling2D((2, 2), name='MaxPooling2D_3')(x)

    # Flatten
    x = Flatten(name='Flatten')(x)
    x = Dropout(0.5, name='Dropout')(x)

    # Bloque extractor
    extractor = Dense(128, activation='relu', name='Dense_Extractor')(x)

    # Bloque decisor - Clasificación
    classification_output = Dense(1, activation='sigmoid', name='Classification_Output')(extractor)

    # Bloque decisor - Regressión (para predecir el centro del tumor)
    regression_output = Dense(2, activation='linear', name='Regression_Output')(extractor)

    model = Model(inputs=input_layer, outputs=[classification_output, regression_output], name='Tumor_Classification_Regression_Model')

    model.compile(optimizer='adam',
                  loss={'Classification_Output': 'binary_crossentropy', 'Regression_Output': 'mse'},
                  metrics={'Classification_Output': [recall_m], 'Regression_Output': 'mae'})

    return model
