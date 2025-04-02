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
        return np.sqrt(np.sum((p1 - p2)**2))

    def fit(self, X_train, y_train):
        """Stores the training data."""
        self._X_train = X_train
        self._y_train = y_train
        print(f"KNN fitted with {self._X_train.shape[0]} training samples.")

    def predict(self, X_test):
        """Predicts labels for test data."""
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Classifier must be fitted before predicting.")

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

    # Optional: Add predict_proba or similar to get scores for FMR/FNMR/ROC
    def predict_proba(self, X_test):
        """Predicts class probabilities (simple version based on neighbor counts)."""
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Classifier must be fitted before predicting.")
        # Note: This is a simplified probability based on neighbor counts
        # For FMR/FNMR, distance-based scores might be better
        all_probas = [self._predict_proba_single(x) for x in X_test]
        return np.array(all_probas)

    def _predict_proba_single(self, x_test):
        """Predicts probabilities for a single test sample."""
        distances = [self._euclidean_distance(x_test, x_train) for x_train in self._X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_neighbor_labels = [self._y_train[i] for i in k_indices]
        
        num_classes = len(np.unique(self._y_train))
        probas = np.zeros(num_classes)
        label_counts = Counter(k_neighbor_labels)
        
        for label, count in label_counts.items():
            probas[label] += count / self.k # Assumes labels are 0 to num_classes-1
            
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
        self.epsilon = 1e-9    # To prevent division by zero variance
        print("Initialized GaussianNBClassifier (from scratch).")

    def fit(self, X_train, y_train):
        """Calculates means, variances, and priors for each class."""
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
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0) + self.epsilon # Add epsilon for numerical stability
            self._priors[idx] = X_c.shape[0] / float(n_samples)
        print(f"GaussianNB fitted with {n_samples} training samples across {n_classes} classes.")


    def predict(self, X_test):
        """Predicts labels for test data."""
        if self._mean is None:
             raise RuntimeError("Classifier must be fitted before predicting.")
        y_pred = [self._predict_single(x) for x in X_test]
        return np.array(y_pred)

    def _predict_single(self, x_test):
        """Predicts the label for a single test sample."""
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx]) # Use log probabilities for numerical stability
            likelihood = np.sum(np.log(self._pdf(idx, x_test)))
            posterior = prior + likelihood
            posteriors.append(posterior)

        # Return class with the highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """Calculates Gaussian Probability Density Function."""
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    # Optional: Get scores for evaluation
    def predict_proba(self, X_test):
        """Predicts class probabilities (log probabilities)."""
        if self._mean is None:
             raise RuntimeError("Classifier must be fitted before predicting.")

        all_posteriors = []
        for x in X_test:
            posteriors = []
            for idx, c in enumerate(self._classes):
                prior = np.log(self._priors[idx])
                likelihood = np.sum(np.log(self._pdf(idx, x)))
                posterior = prior + likelihood
                posteriors.append(posterior)
            # Normalize log probabilities (optional, but good practice if needed elsewhere)
            # max_log_prob = np.max(posteriors)
            # log_probs_shifted = posteriors - max_log_prob
            # probs = np.exp(log_probs_shifted) / np.sum(np.exp(log_probs_shifted))
            all_posteriors.append(posteriors) # Return log posteriors directly

        return np.array(all_posteriors)