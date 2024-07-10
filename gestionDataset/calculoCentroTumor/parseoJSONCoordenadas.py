import os
import json

# =============================================================================
# Proyecto desarrollado como parte del Trabajo de Fin de Estudio
# Máster de Inteligencia Artificial de UNIR
# Autor: Javier López Peña
# Fecha: 10 - 07 - 2024
# =============================================================================

# Nota: Este proyecto ha sido desarrollado íntegramente por el alumno,
# figurando como único autor del código y propietario de sus derechos.

# Script para reescalar los clics guardados en escala 256x256 a 64x64 y pasarlos a fichero txt


def reescala_click(click, original_size, target_size=(64, 64)):
    x, y = click
    scaled_x = int(x * (target_size[0] / original_size[0]))
    scaled_y = int(y * (target_size[1] / original_size[1]))
    return scaled_x, scaled_y

def crear_archivos_txt_from_json(json_file):
    # Cargar el archivo JSON
    with open(json_file, 'r') as f:
        annotations = json.load(f)

    for file_name, data in annotations.items():
        # Obtener la posición del clic y las dimensiones originales
        click = data['click']
        original_size = data['original_size']

        # Reescalar el clic al tamaño 64x64
        scaled_click = reescala_click(click, original_size)

        # Crear el nombre del archivo txt
        txt_file_name = os.path.splitext(file_name)[0] + '.txt'
        txt_file_path = os.path.join(os.path.dirname(json_file), txt_file_name)

        # Escribir las coordenadas en el archivo txt
        with open(txt_file_path, 'w') as txt_file:
            txt_file.write(f"{scaled_click[0]}, {scaled_click[1]}\n")

    print("Archivos TXT creados correctamente.")

if __name__ == "__main__":
    json_file = r'C:\projects\TFM\scripts\calculoCentroTumor\annotations.json'
    crear_archivos_txt_from_json(json_file)
