import os
import shutil
from PIL import Image

# =============================================================================
# Proyecto desarrollado como parte del Trabajo de Fin de Estudio
# Máster de Inteligencia Artificial de UNIR
# Autor: Javier López Peña
# Fecha: 10 - 07 - 2024
# =============================================================================

# Nota: Este proyecto ha sido desarrollado íntegramente por el alumno,
# figurando como único autor del código y propietario de sus derechos.

# Script utilizado para seleccionar o descartar imágenes con tumor

def count_images(folder):
    count = 0
    for root, _, files in os.walk(folder):
        count += len([file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
    return count

def move_images(input_folder, output_folder, not_selected_folder, max_images=20099):
    moved_count = count_images(output_folder)
    total_images_needed = max_images - moved_count

    if total_images_needed <= 0:
        print(f'La carpeta de destino ya tiene {moved_count} imágenes.')
        return

    for root, _, files in os.walk(input_folder):
        total_files = len([file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
        files_seen = 0

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                input_path = os.path.join(root, file)

                # Mostrar la imagen usando Pillow
                image = Image.open(input_path)
                image.show()

                # Decisión basada en la entrada del usuario
                decision = input("Presiona 'm' para mover la imagen, 'n' para no seleccionarla: ").strip().lower()
                files_seen += 1

                if decision == 'm':
                    relative_path = os.path.relpath(root, input_folder)
                    output_dir = os.path.join(output_folder, relative_path)
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, file)
                    shutil.move(input_path, output_path)
                    moved_count += 1
                    print(f'Moved: {file} ({moved_count}/{max_images})')
                else:
                    relative_path = os.path.relpath(root, input_folder)
                    not_selected_dir = os.path.join(not_selected_folder, relative_path)
                    os.makedirs(not_selected_dir, exist_ok=True)
                    not_selected_path = os.path.join(not_selected_dir, file)
                    shutil.move(input_path, not_selected_path)
                    print(f'Not Selected: {file}')

                # Salir si se han movido suficientes imágenes
                if moved_count >= max_images:
                    print(f'Reached the limit of {max_images} images moved.')
                    return

                # Cerrar la imagen
                image.close()
        
        # Informar cuántas fotos quedan por ver en la carpeta de origen
        print(f'Files seen: {files_seen}, Remaining files: {total_files - files_seen}')

if __name__ == "__main__":
    input_folder = r'C:\projects\TFM\gestionDataset\imagenesPreprocesadas\tumor'
    output_folder = r'C:\projects\TFM\gestionDataset\imagenesAntesEtiquetas\tumor'
    not_selected_folder = r'C:\projects\TFM\gestionDataset\imagenesAntesEtiquetas\papelera'
    move_images(input_folder, output_folder, not_selected_folder)
