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
DATASET_PATH = './data/archive/' # ADJUST AS NEEDED
TEST_SIZE = 0.25 # Fraction of data for testing
RANDOM_STATE = 42 # For reproducibility
IMAGE_TARGET_SIZE = (64, 64)

# Feature Extraction Choice
FEATURE_EXTRACTION_METHOD = 'pca' # Options: 'raw', 'pca'
N_PCA_COMPONENTS = 100 # Relevant if method is 'pca'

# Classifier K parameter
KNN_K = 5

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    print("--- Loading Data ---")
    images, labels, label_map = data_loader.load_images(DATASET_PATH)
    # Ensure class_names are sorted consistently corresponding to labels 0, 1, 2...
    class_names = [label_map[i] for i in sorted(label_map.keys())]

    # 2. Preprocess Data (Resize, Grayscale, Flatten)
    print("\n--- Preprocessing Data ---")
    processed_data = preprocessing.preprocess_images(images, target_size=IMAGE_TARGET_SIZE)
    print(f"Preprocessed data shape: {processed_data.shape}")

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
    X_test_scaled = scaler.transform(X_test)

    # 5. Feature Extraction (e.g., PCA) - Fit only on training data!
    print(f"\n--- Applying Feature Extraction ({FEATURE_EXTRACTION_METHOD}) ---")
    if FEATURE_EXTRACTION_METHOD == 'pca':
        # Note: Consider performance without whiten=True if GNB issues persist
        pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE, svd_solver='randomized', whiten=True)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
        print(f"PCA applied. Data shape: {X_train_final.shape}")
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

        # Plot Confusion Matrix (pass name)
        print("Plotting Confusion Matrix...")
        # Pass sorted unique labels from y_test/y_train to ensure correct mapping
        unique_labels_sorted = sorted(np.unique(np.concatenate((y_train, y_test))))
        cm_class_names = [label_map[i] for i in unique_labels_sorted] # Ensure names match used labels
        evaluation.plot_confusion_matrix(y_test, y_pred, class_names=cm_class_names, name=name)

        # Plot ROC Curves (pass name)
        if hasattr(clf, 'predict_proba'):
            try:
                print("Calculating scores for ROC...")
                y_scores = clf.predict_proba(X_test_final)
                print("Plotting ROC Curves (OvR)...")
                # Pass sorted unique labels consistent with training classes for proper mapping
                roc_class_labels = sorted(label_map.keys())
                evaluation.plot_roc_curves_ovr(y_test, y_scores, class_labels=roc_class_labels, name=name)
            except Exception as e:
                print(f"Could not plot ROC curve for {name}: {e}")
        else:
             print(f"ROC curve plotting not available for {name} (needs predict_proba).")

        results[name] = {'accuracy': acc, 'predictions': y_pred}

    # --- Compare Results ---
    print("\n--- Final Results Summary ---")
    for name, result_data in results.items():
        print(f"{name}: Accuracy = {result_data['accuracy']:.4f}")

    print("\n--- Project Outline Complete ---")