import os
from PIL import Image
from rembg import remove
import torchvision.transforms as transforms

# Jalur input dan output
data_path = r"C:\Users\galan\Music\src\d_p"
output_path = r"C:\Users\galan\Music\src\hpsbg"

# Resize ke 256x256
resize_transform = transforms.Resize((512, 512))

# Augmentasi (contoh: flip horizontal & rotasi)
augment_transforms = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(degrees=30),
]

# Loop semua subfolder (kelas penyakit)
for class_folder in os.listdir(data_path):
    class_path = os.path.join(data_path, class_folder)
    if not os.path.isdir(class_path):
        continue

    # Buat folder output-nya kalau belum ada
    class_output_path = os.path.join(output_path, class_folder)
    os.makedirs(class_output_path, exist_ok=True)

    # Loop semua gambar di subfolder
    for filename in os.listdir(class_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Processing {class_folder}/{filename}")

            # Path lengkap
            img_path = os.path.join(class_path, filename)

            # Buka dan resize
            img = Image.open(img_path).convert("RGB")
            img = resize_transform(img)

            # Hapus background
            img_nobg = remove(img)

            # Simpan versi clean
            clean_name = os.path.splitext(filename)[0] + "_clean.png"
            clean_path = os.path.join(class_output_path, clean_name)
            img_nobg.save(clean_path)

            # Augmentasi dari versi clean
            for i, aug in enumerate(augment_transforms):
                aug_img = aug(img_nobg)
                aug_name = os.path.splitext(filename)[0] + f"_aug{i+1}.png"
                aug_path = os.path.join(class_output_path, aug_name)
                aug_img.save(aug_path)
