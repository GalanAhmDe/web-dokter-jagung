import cv2
import numpy as np
import os
import csv
from skimage.feature import local_binary_pattern

# Define paths
grayscale_dataset_path = r"C:\Users\galan\Music\projek_new\gray"  # Path ke dataset grayscale
output_csv_path = r"C:\Users\galan\Music\projek_new\lbp.csv"  # Path untuk menyimpan fitur LBP dalam CSV

# Parameters for LBP
radius = 1  # Radius for LBP
n_points = 8 * radius  # Number of points to consider

# Function to compute LBP and extract features
def compute_lbp_features(image):
    """
    Compute Local Binary Pattern (LBP) for a grayscale image and extract features.
    """
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize the histogram
    return hist

# Open CSV file to write
with open(output_csv_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write header
    header = ["filename", "label"] + [f"LBP_{i}" for i in range(n_points + 2)]
    csv_writer.writerow(header)

    # Get categories (subfolders in grayscale dataset)
    categories = os.listdir(grayscale_dataset_path)

    # Process images
    for category in categories:
        category_path = os.path.join(grayscale_dataset_path, category)

        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)

            # Periksa apakah file adalah gambar (case-insensitive)
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                print(f"File bukan gambar: {image_path}")
                continue

            # Debug: Tampilkan path file
            print(f"Membaca file: {image_path}")

            # Load grayscale image
            grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if grayscale_image is None:
                print(f"Gagal membaca file: {image_path}")
                continue

            # Compute LBP features
            lbp_features = compute_lbp_features(grayscale_image)

            # Write info to CSV
            row = [image_name, category] + lbp_features.tolist()
            csv_writer.writerow(row)

print("✅ Semua gambar telah diproses dengan LBP dan fitur disimpan di CSV!")
print(f"Informasi fitur LBP disimpan di: {output_csv_path}")