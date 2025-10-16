import cv2
import numpy as np
import os

# Define paths
dataset_A_path = r"C:\Users\galan\Music\src\hpsbg"  # Path ke dataset A
dataset_B_path = r"C:\Users\galan\Music\src\gray"  # Path ke dataset B
os.makedirs(dataset_B_path, exist_ok=True)  # Buat folder dataset B jika belum ada

# Debug: Tampilkan struktur folder dataset A
print("Struktur folder dataset A:")
for root, dirs, files in os.walk(dataset_A_path):
    print(f"Folder: {root}")
    print(f"Subfolder: {dirs}")
    print(f"File: {files}")
    print("-" * 50)

# Function to convert image to grayscale
def convert_to_grayscale(image):
    """
    Convert an image to grayscale.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

# Get categories (subfolders in dataset A)
categories = os.listdir(dataset_A_path)

# Process and save grayscale images
for category in categories:
    category_path = os.path.join(dataset_A_path, category)
    save_category_path = os.path.join(dataset_B_path, category)  # Folder untuk kategori di dataset B
    os.makedirs(save_category_path, exist_ok=True)  # Buat folder kategori di dataset B

    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)

        # Periksa apakah file adalah gambar (case-insensitive)
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            print(f"File bukan gambar: {image_path}")
            continue

        # Debug: Tampilkan path file
        print(f"Membaca file: {image_path}")

        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Gagal membaca file: {image_path}")
            continue

        # Convert image to grayscale
        grayscale_image = convert_to_grayscale(original_image)

        # Save the grayscale image
        save_path = os.path.join(save_category_path, image_name)
        cv2.imwrite(save_path, grayscale_image)
        print(f"Grayscale gambar disimpan di: {save_path}")

print("✅ Semua gambar telah dikonversi ke grayscale dan disimpan di dataset B!")