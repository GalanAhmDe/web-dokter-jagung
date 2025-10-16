import cv2
import os

# Define paths
data_path = r"C:\Users\galan\Music\projek_new\hpsbg"  # Folder dataset
output_hsv_path = r"C:\Users\galan\Music\projek_new\hsv"  # Folder untuk menyimpan gambar HSV
os.makedirs(output_hsv_path, exist_ok=True)  # Buat folder jika belum ada

# Function to convert image to HSV
def convert_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

# Function to preprocess and resize image
def preprocess_resize(image_path, size=(512, 512)):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Gagal membaca file: {image_path}")
        return None, None
    resized_image = cv2.resize(original_image, size)
    return original_image, resized_image

# Get categories (subfolders in data_path)
categories = os.listdir(data_path)

# Process and save HSV images
for category in categories:
    category_path = os.path.join(data_path, category)
    save_category_path = os.path.join(output_hsv_path, category)  # Folder untuk kategori
    os.makedirs(save_category_path, exist_ok=True)  # Buat folder kategori

    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)

        # Preprocess image
        original_image, resized_image = preprocess_resize(image_path)
        if original_image is None or resized_image is None:
            continue  # Lewati file jika gagal dibaca

        # Convert to HSV
        hsv_image = convert_to_hsv(resized_image)

        # Save HSV image
        save_path = os.path.join(save_category_path, f"{image_name}")
        cv2.imwrite(save_path, hsv_image)
        print(f"Gambar HSV disimpan di: {save_path}")

print("✅ Semua gambar telah dikonversi ke HSV dan disimpan!")