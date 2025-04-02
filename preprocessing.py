import cv2
import numpy as np

def preprocess_images(images, target_size=(64, 64)):
    """
    Preprocesses a list of images: resize, ensure grayscale, flatten.

    Args:
        images (list): List of images as NumPy arrays (expecting grayscale).
        target_size (tuple): Desired (width, height) for resizing.

    Returns:
        numpy.ndarray: A 2D NumPy array where each row is a flattened, preprocessed image.
    """
    processed_images = []
    for img in images:
        if img is None:
            continue # Skip None images if any slipped through

        # 1. Resize
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        # 2. Ensure Grayscale (already done in load_images, but good practice)
        if len(img_resized.shape) > 2:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_resized

        # 3. (Optional) Normalize pixel values (e.g., to 0-1) - Can help performance
        img_normalized = img_gray / 255.0

        # 4. Flatten
        img_flattened = img_normalized.flatten()
        processed_images.append(img_flattened)

    return np.array(processed_images)

# --- Add functions for HOG if you choose to use it ---
# from skimage.feature import hog
# def extract_hog_features(images, target_size=(64, 64), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
#     hog_features = []
#     for img in images:
#         # Resize and ensure grayscale first (call parts of preprocess_images)
#         img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
#         img_gray = img_resized # Assuming grayscale input
#         features = hog(img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
#                        cells_per_block=cells_per_block, visualize=False, feature_vector=True)
#         hog_features.append(features)
#     return np.array(hog_features)