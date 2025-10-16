import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Untuk tampilan grafik (jika mau)
import matplotlib.pyplot as plt

# Fungsi keanggotaan Gaussian
def fuzzy_membership(hue, center, sigma=10):
    return np.exp(-((hue - center) ** 2) / (2 * sigma ** 2))

# Daftar kategori warna dan center-nya
color_classes = [
    {"name": "Black/Gray", "center": 0},
    {"name": "Red", "center": 0},
    {"name": "Red2", "center": 180},
    {"name": "Orange", "center": 20},
    {"name": "Yellow", "center": 30},
    {"name": "Green", "center": 60},
    {"name": "Cyan", "center": 90},
    {"name": "Blue", "center": 120},
    {"name": "Magenta", "center": 150},
    {"name": "Pink", "center": 170},
]

# ======================
# 1. Buka dialog unggah
# ======================
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Pilih gambar RGB")

if not file_path:
    print("❌ Tidak ada file dipilih.")
    exit()

print(f"📂 Gambar dipilih: {file_path}")

# ======================
# 2. Baca dan resize gambar
# ======================
bgr = cv2.imread(file_path)
bgr = cv2.resize(bgr, (512, 512))  # Pastikan ukurannya 512x512
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
hue = hsv[:, :, 0]

print(f"✅ Ukuran gambar: {hue.shape} ({hue.size} piksel)")

# ================================
# 3. Cuplikan 5 piksel pertama
# ================================
hue_flat = hue.flatten()
sample_hue = hue_flat[:5]
print("\n🟡 Contoh nilai Hue dari 5 piksel pertama:", sample_hue.tolist())

# =================================
# 4. Hitung FCH (per warna manual)
# =================================
sigma = 10
fch_values = []

for color in color_classes:
    center = color["center"]
    mu_all = fuzzy_membership(hue, center, sigma)  # Untuk semua piksel
    mean_mu = np.mean(mu_all)
    fch_values.append(mean_mu)

    # Tampilkan perhitungan untuk 5 piksel pertama
    print(f"\n🔸 Warna: {color['name']} (center={center})")
    for idx, h in enumerate(sample_hue):
        mu = fuzzy_membership(h, center, sigma)
        print(f"  h = {h:3} → μ = exp(-((h - {center})²)/(2×{sigma}²)) = {mu:.4f}")
    print(f"  ➤ Rata-rata keanggotaan (FCH_{color['name']}) = {mean_mu:.4f}")

# ====================================
# 5. Normalisasi histogram FCH
# ====================================
total = sum(fch_values)
fch_norm = [v / total for v in fch_values]

print("\n📊 Hasil Akhir: Fuzzy Color Histogram (dinormalisasi):")
for color, value in zip(color_classes, fch_norm):
    print(f"  {color['name']:10s} : {value:.4f}")
