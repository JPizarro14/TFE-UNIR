import os
import json
from PIL import Image, ImageTk
import tkinter as tk

# =============================================================================
# Proyecto desarrollado como parte del Trabajo de Fin de Estudio
# Máster de Inteligencia Artificial de UNIR
# Autor: Javier López Peña
# Fecha: 10 - 07 - 2024
# =============================================================================

# Nota: Este proyecto ha sido desarrollado íntegramente por el alumno,
# figurando como único autor del código y propietario de sus derechos.

# Script utilizado para abrir la imagen con tumor ampliada, clicar en ella y utilizando 
# la posición donde se ha hecho click guardarlo en un json

def on_click(event, clicks, img_size, root):
    x, y = event.x, event.y
    print(f"Clicked at: ({x}, {y})")
    scaled_x = int(x * (64 / img_size[0]))
    scaled_y = int(y * (64 / img_size[1]))
    clicks.append((scaled_x, scaled_y))
    root.destroy()  # Cierra la ventana después del clic

def annotate_images(image_folder, output_json_file, display_size=(1024, 1024)):
    if os.path.exists(output_json_file):
        with open(output_json_file, 'r') as f:
            annotations = json.load(f)
    else:
        annotations = {}

    for root_dir, _, files in os.walk(image_folder):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                if file_name in annotations:
                    continue  # Skip files already annotated

                input_path = os.path.join(root_dir, file_name)

                # Mostrar la imagen usando Tkinter y Pillow
                img = Image.open(input_path)
                original_size = img.size
                img = img.resize(display_size, Image.Resampling.LANCZOS)
                root = tk.Tk()
                root.title(file_name)
                tk_img = ImageTk.PhotoImage(img)
                image_label = tk.Label(root, image=tk_img)
                image_label.pack()

                clicks = []
                image_label.bind("<Button-1>", lambda event, clks=clicks, img_size=original_size, rt=root: on_click(event, clks, img_size, rt))
                root.mainloop()

                # Guardar las coordenadas del clic
                if clicks:
                    annotations[file_name] = {
                        'click': clicks[0],
                        'original_size': original_size,
                        'scaled_size': (64, 64)
                    }

                    # Guardar la imagen reescalada a 64x64
                    img_64 = img.resize((64, 64), Image.Resampling.LANCZOS)
                    img_64.save(os.path.join(root_dir, f"64x64_{file_name}"))

    # Guardar las anotaciones en un archivo JSON
    with open(output_json_file, 'w') as f:
        json.dump(annotations, f, indent=4)
    print(f"Annotations saved to {output_json_file}")

if __name__ == "__main__":
    input_folder = r'C:\projects\TFM\gestionDataset\imagenesAntesEtiquetas\tumor'
    output_json = r'C:\projects\TFM\scripts\calculoCentroTumor\annotations.json'
    annotate_images(input_folder, output_json)
