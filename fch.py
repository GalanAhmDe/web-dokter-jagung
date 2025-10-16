import cv2
import numpy as np
import os
import pandas as pd

# Define paths
data_path = r"C:\Users\galan\Music\src\hsv"  # Folder gambar HSV
output_csv_path = r"C:\Users\galan\Music\src\fch.csv"  # File CSV untuk menyimpan fitur FCH

# Function to apply fuzzy color classification
def fuzzy_color_classification(hsv_image):
    h, s, v = cv2.split(hsv_image)
    num_classes = 10  # Jumlah kelas warna
    fuzzy_histogram = np.zeros(num_classes)  # Inisialisasi fuzzy histogram

    # Define fuzzy membership functions for each color class
    def fuzzy_membership(hue, center, sigma):
        return np.exp(-((hue - center) ** 2) / (2 * sigma ** 2))

    # Define color classes (centers and sigma for fuzzy membership)
    color_classes = [
        {"name": "Black/Gray", "center": 0, "sigma": 10},  # Black/Gray (Low V)
        {"name": "Red", "center": 0, "sigma": 10},         # Red (0-10)
        {"name": "Red", "center": 180, "sigma": 10},       # Red (170-180)
        {"name": "Orange", "center": 20, "sigma": 10},     # Orange
        {"name": "Yellow", "center": 30, "sigma": 10},     # Yellow
        {"name": "Green", "center": 60, "sigma": 10},      # Green
        {"name": "Cyan", "center": 90, "sigma": 10},       # Cyan
        {"name": "Blue", "center": 120, "sigma": 10},      # Blue
        {"name": "Magenta", "center": 150, "sigma": 10},    # Magenta
        {"name": "Pink", "center": 170, "sigma": 10}       # Pink
    ]

    # Calculate fuzzy membership for each pixel
    for i, color in enumerate(color_classes):
        fuzzy_histogram[i] = np.mean(fuzzy_membership(h, color["center"], color["sigma"]))

    # Normalize fuzzy histogram
    fuzzy_histogram /= np.sum(fuzzy_histogram)

    return fuzzy_histogram

# Collecting FCH features along with filenames and labels
fch_data = []

# Get categories (subfolders in data_path)
categories = os.listdir(data_path)

# Process and extract FCH features
for category in categories:
    category_path = os.path.join(data_path, category)

    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)

        # Load HSV image
        hsv_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if hsv_image is None:
            print(f"Gagal membaca file: {image_path}")
            continue

        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)

        # Apply fuzzy color classification
        fch_features = fuzzy_color_classification(hsv_image)

        # Append filename, label, and FCH features
        fch_data.append([image_name, category] + list(fch_features))

# Save FCH features to CSV
columns = ['filename', 'label'] + [f'fch_feature_{i}' for i in range(len(fch_features))]
df = pd.DataFrame(fch_data, columns=columns)
df.to_csv(output_csv_path, index=False)

print(f"✅ Ekstraksi fitur FCH selesai dan disimpan di: {output_csv_path}")