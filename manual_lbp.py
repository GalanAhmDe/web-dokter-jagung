import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Fungsi untuk membuka file gambar
def pilih_gambar():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Pilih gambar", 
                                           filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return file_path

# Thresholding
def thresholding(center, neighbors):
    return [1 if n >= center else 0 for n in neighbors]

# Hitung transisi
def calculate_uniform_transitions(binary_pattern):
    transitions = 0
    for i in range(len(binary_pattern)):
        if binary_pattern[i] != binary_pattern[(i + 1) % len(binary_pattern)]:
            transitions += 1
    return transitions

# Hitung label uniform
def get_uniform_label(binary_pattern):
    transitions = calculate_uniform_transitions(binary_pattern)
    if transitions <= 2:
        return sum(binary_pattern)
    else:
        return 58

# Ambil 8 tetangga
def get_neighbors(img, x, y):
    return [
        img[x-1, y-1], img[x-1, y], img[x-1, y+1],
        img[x, y+1], img[x+1, y+1], img[x+1, y],
        img[x+1, y-1], img[x, y-1]
    ]

# Proses satu piksel pusat dan tampilkan info jika perlu
def proses_lbp(img, x, y, tampilkan=False):
    center = img[x, y]
    neighbors = get_neighbors(img, x, y)
    biner = thresholding(center, neighbors)
    transisi = calculate_uniform_transitions(biner)
    label = get_uniform_label(biner)

    if tampilkan:
        print(f"\n📍 Posisi pusat: ({x},{y})")
        print(f"Nilai pusat       : {center}")
        print(f"Tetangga          : {neighbors}")
        print(f"Thresholding      : {biner}")
        print(f"Jumlah transisi   : {transisi}")
        print(f"Label Uniform     : {label}")

    return label

# === MULAI ===
file_path = pilih_gambar()
if not file_path:
    print("❌ Tidak ada gambar dipilih.")
else:
    print(f"✅ Gambar terpilih: {file_path}")

    # Baca gambar grayscale
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if img.shape != (512, 512):
        print(f"⚠️ Ukuran gambar bukan 512x512, melainkan {img.shape}. Resize otomatis.")
        img = cv2.resize(img, (512, 512))

    plt.imshow(img, cmap='gray')
    plt.title("Gambar 512x512")
    plt.axis('off')
    plt.show()

    # Proses semua piksel pusat
    labels = []
    for x in range(1, 511):
        for y in range(1, 511):
            tampilkan = (x == 1 and y == 1)  # hanya tampilkan piksel pusat pertama
            label = proses_lbp(img, x, y, tampilkan=tampilkan)
            labels.append(label)

    # Histogram dan normalisasi
    hist = [0] * 59
    for label in labels:
        hist[label] += 1

    total_pixels = len(labels)
    hist_norm = [h / total_pixels for h in hist]

    print("\n==============================")
    print(f"📌 Total piksel pusat diproses: {total_pixels}")
    print("==============================")

    print("\n📊 Histogram Frekuensi LBP Uniform (Label 0–58):")
    for i in range(59):
        print(f"Label {i:2d} : {hist[i]}")

    print("\n📊 Histogram Normalisasi:")
    for i in range(59):
        print(f"Label {i:2d} : {hist_norm[i]:.6f}")
