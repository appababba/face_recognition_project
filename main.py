import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Feature scaling is often crucial
from sklearn.decomposition import PCA # Example Feature Extraction
# from skimage.feature import hog # If using HOG

# Import your modules
import data_loader
import preprocessing
import classifiers
import evaluation

# --- Configuration ---
DATASET_PATH = './data/cmu_face_images' # ADJUST AS NEEDED
# DATASET_PATH = './data/some_kaggle_dataset' # ADJUST AS NEEDED
TEST_SIZE = 0.25 # Fraction of data for testing
RANDOM_STATE = 42 # For reproducibility
IMAGE_TARGET_SIZE = (64, 64)

# Feature Extraction Choice
# Options: 'raw', 'pca' # Add 'hog' if implemented
FEATURE_EXTRACTION_METHOD = 'pca'
N_PCA_COMPONENTS = 100 # Relevant if method is 'pca'

# Classifier K parameter
KNN_K = 5

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    print("--- Loading Data ---")
    images, labels, label_map = data_loader.load_images(DATASET_PATH)
    class_names = [label_map[i] for i in sorted(label_map.keys())] # Get class names in order

    # 2. Preprocess Data (Resize, Grayscale, Flatten)
    print("\n--- Preprocessing Data ---")
    # If using HOG, you might need a separate preprocessing step before HOG extraction
    processed_data = preprocessing.preprocess_images(images, target_size=IMAGE_TARGET_SIZE)
    print(f"Preprocessed data shape: {processed_data.shape}") # (n_samples, n_features_flattened)

    # 3. Split Data into Training and Testing Sets
    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")

    # 4. Feature Scaling (Important!) - Fit only on training data!
    print("\n--- Scaling Features ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Use the same scaler fitted on train data

    # 5. Feature Extraction (Optional, e.g., PCA) - Fit only on training data!
    print(f"\n--- Applying Feature Extraction ({FEATURE_EXTRACTION_METHOD}) ---")
    if FEATURE_EXTRACTION_METHOD == 'pca':
        pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE, svd_solver='randomized', whiten=True)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
        print(f"PCA applied. Data shape: {X_train_final.shape}")
    # elif FEATURE_EXTRACTION_METHOD == 'hog':
        # Need to apply HOG *before* split/scaling potentially, or adjust logic here
        # X_train_final = extract_hog_features(X_train_images...) # Requires access to images before flattening/scaling
        # X_test_final = extract_hog_features(X_test_images...)
        # print(f"HOG features extracted. Data shape: {X_train_final.shape}")
        # pass # Placeholder
    elif FEATURE_EXTRACTION_METHOD == 'raw':
        X_train_final = X_train_scaled # Use scaled raw pixels
        X_test_final = X_test_scaled
        print("Using scaled raw pixel features.")
    else:
        raise ValueError(f"Unknown feature extraction method: {FEATURE_EXTRACTION_METHOD}")

    # --- Train and Evaluate Classifiers ---

    classifiers_to_run = {
        "KNN (Scratch)": classifiers.KNNClassifier(k=KNN_K),
        "GaussianNB (Scratch)": classifiers.GaussianNBClassifier()
        # Add sklearn versions for comparison if needed:
        # "KNN (Sklearn)": KNeighborsClassifier(n_neighbors=KNN_K),
        # "GaussianNB (Sklearn)": GaussianNB()
    }

    results = {}

    for name, clf in classifiers_to_run.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train_final, y_train)

        print(f"\n--- Evaluating {name} ---")
        y_pred = clf.predict(X_test_final)

        # Calculate Accuracy
        acc = evaluation.calculate_accuracy(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        # Plot Confusion Matrix
        print("Plotting Confusion Matrix...")
        evaluation.plot_confusion_matrix(y_test, y_pred, class_names=class_names)

        # Plot ROC Curves (Requires predict_proba)
        if hasattr(clf, 'predict_proba'):
            try:
                print("Calculating scores for ROC...")
                # Note: Naive Bayes might return log_proba, kNN simple proba
                # Ensure scores are suitable for roc_curve function (probabilities of positive class for OvR)
                y_scores = clf.predict_proba(X_test_final)
                print("Plotting ROC Curves (OvR)...")
                evaluation.plot_roc_curves_ovr(y_test, y_scores, class_labels=sorted(label_map.keys()))
            except Exception as e:
                print(f"Could not plot ROC curve for {name}: {e}")
        else:
             print(f"ROC curve plotting not available for {name} (needs predict_proba).")

        results[name] = {'accuracy': acc, 'predictions': y_pred} # Store results if needed

    # --- Compare Results ---
    print("\n--- Final Results Summary ---")
    for name, result_data in results.items():
        print(f"{name}: Accuracy = {result_data['accuracy']:.4f}")

    print("\n--- Project Outline Complete ---")