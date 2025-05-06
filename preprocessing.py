import cv2
import numpy as np

def preprocess_images(images, target_size=(64, 64)):
    """
preprocess list of images, returns 2d numpy array where each row is flattened image

    """
    processed_images = []
    for img in images: # skip any failed loads
        if img is None:
            continue 

        # resize image to target dimensions
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        # if 3-channel, convert to single-channel grayscale
        if len(img_resized.shape) > 2:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_resized

# normalize pixel values to [0, 1]
        img_normalized = img_gray / 255.0

        # flatten
        img_flattened = img_normalized.flatten()
        processed_images.append(img_flattened)

    return np.array(processed_images)

