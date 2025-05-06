import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # feature scaling
from sklearn.decomposition import PCA  # pca feature extraction

# custom modules
import data_loader
import preprocessing
import classifiers
import evaluation


# --- config ---
dataset_path = './data/archive/'  # where to load data from
test_size = 0.25  # test set fraction
random_state = 42  # for reproducible splits
image_target_size = (64, 64)  # resize dims

feature_extraction_method = 'pca'  # 'raw' or 'pca'
n_pca_components = 100  # num components if pca
knn_k = 5  # k for knn


if __name__ == "__main__":
    # load data
    print("--- Loading Data ---")
    images, labels, label_map = data_loader.load_images(dataset_path)
    class_names = [label_map[i] for i in sorted(label_map.keys())]

    # preprocess data
    print("\n--- Preprocessing Data ---")
    processed_data = preprocessing.preprocess_images(images, target_size=image_target_size)
    print(f"Preprocessed data shape: {processed_data.shape}")

    # split data into testing and training
    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        processed_data, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")

    # feature scaling
    print("\n--- Scaling Features ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # feature extraction
    print(f"\n--- Applying Feature Extraction ({feature_extraction_method}) ---")
    if feature_extraction_method == 'pca':
        pca = PCA(n_components=n_pca_components, random_state=random_state, svd_solver='randomized', whiten=True)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
        print(f"PCA applied. Data shape: {X_train_final.shape}")
    elif feature_extraction_method == 'raw':
        X_train_final = X_train_scaled # use scaled raw pixels
        X_test_final = X_test_scaled
        print("Using scaled raw pixel features.")
    else:
        raise ValueError(f"Unknown feature extraction method: {feature_extraction_method}")

    # train and test classifiers
    classifiers_to_run = {
        "KNN (Scratch)": classifiers.KNNClassifier(k=knn_k),
        "GaussianNB (Scratch)": classifiers.GaussianNBClassifier()
    }

    results = {}

    for name, clf in classifiers_to_run.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train_final, y_train)

        print(f"\n--- Evaluating {name} ---")
        y_pred = clf.predict(X_test_final)

        acc = evaluation.calculate_accuracy(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        # plot confusion matrix
        print("Plotting Confusion Matrix...")
        # pass sorted unique labels from y_test/y_train
        unique_labels_sorted = sorted(np.unique(np.concatenate((y_train, y_test))))
        cm_class_names = [label_map[i] for i in unique_labels_sorted] # names match labels
        evaluation.plot_confusion_matrix(y_test, y_pred, class_names=cm_class_names, name=name)

        # plot ROC curves
        if hasattr(clf, 'predict_proba'):
            try:
                print("Calculating scores for ROC...")
                y_scores = clf.predict_proba(X_test_final)
                print("Plotting ROC Curves (OvR)...")
                # pass sorted labels consistent with training classes for proper mapping
                roc_class_labels = sorted(label_map.keys())
                evaluation.plot_roc_curves_ovr(y_test, y_scores, class_labels=roc_class_labels, name=name)
            except Exception as e:
                print(f"Could not plot ROC curve for {name}: {e}")
        else:
            print(f"ROC curve plotting not available for {name} (needs predict_proba).")

        results[name] = {'accuracy': acc, 'predictions': y_pred}

    # compare results
    print("\n--- Final Results Summary ---")
    for name, result_data in results.items():
        print(f"{name}: Accuracy = {result_data['accuracy']:.4f}")

    print("\n--- Project Outline Complete ---")