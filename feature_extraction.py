import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image
from rembg import remove
import torchvision.transforms as transforms

def preprocess_resize(image_path, size=(512, 512)):
    """
    Identical to preprocess.py:
    1. Open with PIL
    2. Resize to 256x256
    3. Remove background
    """
    try:
        # Buka gambar dengan PIL
        img = Image.open(image_path).convert("RGB")
        
        # Resize
        img = transforms.Resize(size)(img)
        
        # Hapus background
        img_nobg = remove(img)
        
        # Konversi ke format OpenCV (BGR)
        processed_image = cv2.cvtColor(np.array(img_nobg), cv2.COLOR_RGB2BGR)
        
        # Baca juga original image dengan OpenCV untuk return
        original_image = cv2.imread(image_path)
        
        return original_image, processed_image
    
    except Exception as e:
        print(f"Error preprocessing {image_path}: {str(e)}")
        return None, None

def convert_to_grayscale(image):
    """Identical to grayscale.py"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def compute_lbp_features(image, radius=1, n_points=8):
    """Identical to lbp.py"""
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def convert_to_hsv(image):
    """Identical to hsv.py"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def fuzzy_color_classification(hsv_image):
    """Identical to fch.py"""
    h, s, v = cv2.split(hsv_image)
    num_classes = 10
    fuzzy_histogram = np.zeros(num_classes)
    
    def fuzzy_membership(hue, center, sigma):
        return np.exp(-((hue - center) ** 2) / (2 * sigma ** 2))
    
    color_classes = [
        {"name": "Black/Gray", "center": 0, "sigma": 10},
        {"name": "Red", "center": 0, "sigma": 10},
        {"name": "Red", "center": 180, "sigma": 10},
        {"name": "Orange", "center": 20, "sigma": 10},
        {"name": "Yellow", "center": 30, "sigma": 10},
        {"name": "Green", "center": 60, "sigma": 10},
        {"name": "Cyan", "center": 90, "sigma": 10},
        {"name": "Blue", "center": 120, "sigma": 10},
        {"name": "Magenta", "center": 150, "sigma": 10},
        {"name": "Pink", "center": 170, "sigma": 10}
    ]
    
    for i, color in enumerate(color_classes):
        fuzzy_histogram[i] = np.mean(fuzzy_membership(h, color["center"], color["sigma"]))
    
    fuzzy_histogram /= np.sum(fuzzy_histogram)
    return fuzzy_histogram

def extract_features(image_path, debug=False):
    """
    Complete feature extraction pipeline with debug option
    Returns: combined LBP and FCH features
    """
    # Step 1: Preprocessing
    original_image, processed_image = preprocess_resize(image_path)
    if original_image is None or processed_image is None:
        return None

    # Step 2: Grayscale conversion
    grayscale_image = convert_to_grayscale(processed_image)
    
    # Step 3: LBP feature extraction
    lbp_features = compute_lbp_features(grayscale_image)
    
    # Step 4: HSV conversion
    hsv_image = convert_to_hsv(processed_image)
    
    # Step 5: FCH feature extraction
    fch_features = fuzzy_color_classification(hsv_image)

    if debug:
        print("\n=== DEBUG FEATURE EXTRACTION ===")
        print("LBP Features:", np.round(lbp_features, 4))
        print("FCH Features:", np.round(fch_features, 4))
        print("===============================\n")

    return np.hstack([lbp_features, fch_features])