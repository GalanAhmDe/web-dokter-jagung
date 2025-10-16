from rembg import remove
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os

root = tk.Tk()
root.withdraw()  

input_path = filedialog.askopenfilename(
    title="Pilih gambar",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if input_path:
    input_image = Image.open(input_path)
    output_image = remove(input_image)

    output_path = os.path.join(os.path.dirname(input_path), 'output.jpg')
    output_image.save(output_path)

    print(f"Gambar berhasil diproses dan disimpan di: {output_path}")
else:
    print("Tidak ada file yang dipilih.")
