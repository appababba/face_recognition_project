import os
import cv2
import numpy as np
from glob import glob # For finding files easily

def load_images(dataset_path, file_extensions=('.pgm', '.jpg', '.png', '.jpeg')):
    """
    Loads images from a dataset directory structured with subfolders per class/person.

    Args:
        dataset_path (str): Path to the root directory of the dataset.
        file_extensions (tuple): Tuple of allowed image file extensions.

    Returns:
        tuple: A tuple containing:
            - images (list): A list of loaded images as NumPy arrays.
            - labels (list): A list of corresponding integer labels.
            - label_map (dict): A dictionary mapping integer labels to class/person names (subfolder names).
    """
    images = []
    labels = []
    label_map = {}
    current_label = 0

    print(f"Loading images from: {dataset_path}")
    # Iterate through subdirectories (each assumed to be a class/person)
    for person_name in sorted(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            if person_name not in label_map.values():
                 label_map[current_label] = person_name
            else: # Find existing label if already seen
                 current_label = [k for k, v in label_map.items() if v == person_name][0]

            print(f"  Loading images for: {person_name} (Label: {current_label})")
            image_files = []
            for ext in file_extensions:
                image_files.extend(glob(os.path.join(person_dir, f'*{ext}')))

            for image_path in image_files:
                # Load image in grayscale directly
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(current_label)
                else:
                    print(f"    Warning: Could not load image {image_path}")

            # Increment label only if we actually processed a new person
            if person_name == label_map[current_label]:
                 current_label += 1 # Move to next label for the next person folder

    if not images:
         raise ValueError(f"No images found in {dataset_path}. Check path and subfolder structure.")

    print(f"Loaded {len(images)} images from {len(label_map)} classes.")
    return images, np.array(labels), label_map