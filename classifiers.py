import numpy as np
from collections import Counter

class KNNClassifier:
    """
    K-Nearest Neighbors classifier implemented from scratch.
    """
    def __init__(self, k=3):
        """Initializes the KNN classifier."""
        assert k >= 1, "k must be at least 1"
        self.k = k
        self._X_train = None
        self._y_train = None
        print(f"Initialized KNNClassifier (from scratch) with k={self.k}")

    def _euclidean_distance(self, p1, p2):
        """Calculates Euclidean distance between two points."""
        # Ensure inputs are numpy arrays for vectorized operations
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        return np.sqrt(np.sum((p1 - p2)**2))

    def fit(self, X_train, y_train):
        """Stores the training data."""
        self._X_train = np.asarray(X_train)
        self._y_train = np.asarray(y_train)
        print(f"KNN fitted with {self._X_train.shape[0]} training samples.")

    def predict(self, X_test):
        """Predicts labels for test data."""
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Classifier must be fitted before predicting.")
        X_test = np.asarray(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x_test):
        """Predicts the label for a single test sample."""
        # 1. Calculate distances to all training points
        distances = [self._euclidean_distance(x_test, x_train) for x_train in self._X_train]

        # 2. Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # 3. Get labels of these neighbors
        k_neighbor_labels = [self._y_train[i] for i in k_indices]

        # 4. Return the most common class label (majority vote)
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

    def predict_proba(self, X_test):
        """Predicts class probabilities (simple version based on neighbor counts)."""
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Classifier must be fitted before predicting.")
        X_test = np.asarray(X_test)
        # Determine the number of classes based on unique labels in training data
        # Ensure classes are sorted so indexing works correctly
        self._unique_classes = sorted(np.unique(self._y_train))
        num_classes = len(self._unique_classes)
        class_to_index = {label: idx for idx, label in enumerate(self._unique_classes)}

        all_probas = [self._predict_proba_single(x, num_classes, class_to_index) for x in X_test]
        return np.array(all_probas)

    def _predict_proba_single(self, x_test, num_classes, class_to_index):
        """Predicts probabilities for a single test sample."""
        distances = [self._euclidean_distance(x_test, x_train) for x_train in self._X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self._y_train[i] for i in k_indices]

        probas = np.zeros(num_classes)
        label_counts = Counter(k_neighbor_labels)

        for label, count in label_counts.items():
            if label in class_to_index: # Check if label exists in training classes
                 probas[class_to_index[label]] += count / self.k

        return probas


class GaussianNBClassifier:
    """
    Gaussian Naive Bayes classifier implemented from scratch.
    """
    def __init__(self):
        """Initializes the GaussianNB classifier."""
        self._classes = None
        self._mean = None      # Shape: (n_classes, n_features)
        self._var = None       # Shape: (n_classes, n_features)
        self._priors = None    # Shape: (n_classes,)
        # Epsilon for variance calculation (increased stability)
        self.variance_epsilon = 1e-6
        # Epsilon for preventing log(0) (can be smaller)
        self.log_epsilon = 1e-12
        print("Initialized GaussianNBClassifier (from scratch).")

    def fit(self, X_train, y_train):
        """Calculates means, variances, and priors for each class."""
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        n_samples, n_features = X_train.shape
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)

        # Initialize parameters
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self._classes):
            X_c = X_train[y_train == c] # Get samples belonging to class c
            if X_c.shape[0] == 0: # Handle case where a class might not be in train split
                continue
            self._mean[idx, :] = X_c.mean(axis=0)
            # Add epsilon for numerical stability DURING VARIANCE CALCULATION
            self._var[idx, :] = X_c.var(axis=0) + self.variance_epsilon
            self._priors[idx] = X_c.shape[0] / float(n_samples)
        print(f"GaussianNB fitted with {n_samples} training samples across {n_classes} classes.")


    def predict(self, X_test):
        """Predicts labels for test data."""
        if self._mean is None:
             raise RuntimeError("Classifier must be fitted before predicting.")
        X_test = np.asarray(X_test)
        y_pred = [self._predict_single(x) for x in X_test]
        return np.array(y_pred)

    def _predict_single(self, x_test):
        """Predicts the label for a single test sample."""
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            # Handle case where class might have had no training samples
            if self._priors[idx] == 0:
                 posteriors.append(-np.inf) # Assign very low probability
                 continue

            prior = np.log(self._priors[idx]) # Use log probabilities for numerical stability

            # Calculate PDF values
            pdf_values = self._pdf(idx, x_test)
            # Clip values to prevent log(0) using log_epsilon
            safe_pdf_values = np.maximum(pdf_values, self.log_epsilon)
            # Calculate log likelihood
            likelihood = np.sum(np.log(safe_pdf_values))

            posterior = prior + likelihood
            posteriors.append(posterior)

        # Return class with the highest posterior probability
        # Check if all posteriors are -inf (e.g., if test point is extreme outlier for all)
        if np.all(np.isneginf(posteriors)):
            # Handle this case - perhaps predict the most probable class overall?
            # Or return a special indicator? For now, predict based on priors.
             print(f"Warning: Test sample resulted in -inf posterior for all classes. Predicting based on prior.")
             return self._classes[np.argmax(self._priors)]

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """Calculates Gaussian Probability Density Function."""
        mean = self._mean[class_idx]
        var = self._var[class_idx] # Already includes variance_epsilon
        # Handle potential zero variance again just in case (though epsilon should prevent it)
        var = np.maximum(var, 1e-15) # Safety net
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        # Avoid division by zero in denominator
        # This is less likely now due to epsilon in variance, but good practice
        # If denominator is near zero, pdf is near zero anyway unless numerator is also zero
        pdf = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        return pdf


    def predict_proba(self, X_test):
        """Predicts class probabilities (log probabilities)."""
        if self._mean is None:
             raise RuntimeError("Classifier must be fitted before predicting.")
        X_test = np.asarray(X_test)
        all_posteriors = []
        for x in X_test:
            posteriors = []
            for idx, c in enumerate(self._classes):
                # Handle case where class might have had no training samples
                if self._priors[idx] == 0:
                     posteriors.append(-np.inf) # Assign very low probability
                     continue

                prior = np.log(self._priors[idx])

                # Calculate PDF values
                pdf_values = self._pdf(idx, x)
                # Clip values to prevent log(0) using log_epsilon
                safe_pdf_values = np.maximum(pdf_values, self.log_epsilon)
                # Calculate log likelihood
                likelihood = np.sum(np.log(safe_pdf_values))

                posterior = prior + likelihood
                posteriors.append(posterior)
            all_posteriors.append(posteriors) # Return log posteriors directly

        return np.array(all_posteriors)