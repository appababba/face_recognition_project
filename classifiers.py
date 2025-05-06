import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        assert k >= 1, "k must be at least 1" # ensure k is valid
        self.k = k # number of neighbors
        self._X_train = None # training data features
        self._y_train = None # training data labels
        print(f"Initialized KNNClassifier (from scratch) with k={self.k}")

    def _euclidean_distance(self, p1, p2):
        p1 = np.asarray(p1) # convert to numpy array
        p2 = np.asarray(p2) # convert to numpy array
        return np.sqrt(np.sum((p1 - p2)**2)) # calculate euclidean distance

    def fit(self, X_train, y_train):
        self._X_train = np.asarray(X_train) # store training features
        self._y_train = np.asarray(y_train) # store training labels
        print(f"KNN fitted with {self._X_train.shape[0]} training samples.")

    def predict(self, X_test):
        if self._X_train is None or self._y_train is None: # check if fitted
            raise RuntimeError("Classifier must be fitted before predicting.")
        X_test = np.asarray(X_test) # convert to numpy array
        predictions = [self._predict_single(x) for x in X_test] # predict for each test sample
        return np.array(predictions)

    def _predict_single(self, x_test):
        distances = [self._euclidean_distance(x_test, x_train) for x_train in self._X_train] # distances to all train points
        k_indices = np.argsort(distances)[:self.k] # indices of k nearest neighbors
        k_neighbor_labels = [self._y_train[i] for i in k_indices] # labels of these neighbors
        most_common = Counter(k_neighbor_labels).most_common(1) # majority vote
        return most_common[0][0]

    def predict_proba(self, X_test):
        if self._X_train is None or self._y_train is None: # check if fitted
            raise RuntimeError("Classifier must be fitted before predicting.")
        X_test = np.asarray(X_test) # convert to numpy array
        self._unique_classes = sorted(np.unique(self._y_train)) # get sorted unique classes
        num_classes = len(self._unique_classes) # number of classes
        class_to_index = {label: idx for idx, label in enumerate(self._unique_classes)} # map class label to index
        all_probas = [self._predict_proba_single(x, num_classes, class_to_index) for x in X_test] # get probas for each sample
        return np.array(all_probas)

    def _predict_proba_single(self, x_test, num_classes, class_to_index):
        distances = [self._euclidean_distance(x_test, x_train) for x_train in self._X_train] # distances to all train points
        k_indices = np.argsort(distances)[:self.k] # indices of k nearest
        k_neighbor_labels = [self._y_train[i] for i in k_indices] # labels of these neighbors
        probas = np.zeros(num_classes) # initialize probabilities
        label_counts = Counter(k_neighbor_labels) # count neighbor labels
        for label, count in label_counts.items():
            if label in class_to_index: # check if label is a known class
                probas[class_to_index[label]] += count / self.k # calculate probability for class
        return probas


class GaussianNBClassifier:
    def __init__(self):
        self._classes = None # unique classes
        self._mean = None # mean per class per feature
        self._var = None # variance per class per feature
        self._priors = None # prior probability of each class
        self.variance_epsilon = 1e-6 # for variance calculation stability
        self.log_epsilon = 1e-12 # for preventing log(0)
        print("Initialized GaussianNBClassifier (from scratch).")

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train) # convert to numpy array
        y_train = np.asarray(y_train) # convert to numpy array
        n_samples, n_features = X_train.shape # get data dimensions
        self._classes = np.unique(y_train) # find unique classes
        n_classes = len(self._classes) # number of classes

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64) # init means
        self._var = np.zeros((n_classes, n_features), dtype=np.float64) # init variances
        self._priors = np.zeros(n_classes, dtype=np.float64) # init priors

        for idx, c in enumerate(self._classes): # for each class
            X_c = X_train[y_train == c] # samples for class c
            if X_c.shape[0] == 0: # if no samples for this class
                continue
            self._mean[idx, :] = X_c.mean(axis=0) # calculate mean
            self._var[idx, :] = X_c.var(axis=0) + self.variance_epsilon # calculate variance with epsilon
            self._priors[idx] = X_c.shape[0] / float(n_samples) # calculate prior
        print(f"GaussianNB fitted with {n_samples} training samples across {n_classes} classes.")


    def predict(self, X_test):
        if self._mean is None: # check if fitted
             raise RuntimeError("Classifier must be fitted before predicting.")
        X_test = np.asarray(X_test) # convert to numpy array
        y_pred = [self._predict_single(x) for x in X_test] # predict for each test sample
        return np.array(y_pred)

    def _predict_single(self, x_test):
        posteriors = [] # list to store posterior probabilities
        for idx, c in enumerate(self._classes): # for each class
            if self._priors[idx] == 0: # if class had no training samples
                posteriors.append(-np.inf) # assign very low probability
                continue
            prior = np.log(self._priors[idx]) # log of prior
            pdf_values = self._pdf(idx, x_test) # calculate pdf
            safe_pdf_values = np.maximum(pdf_values, self.log_epsilon) # clip to avoid log(0)
            likelihood = np.sum(np.log(safe_pdf_values)) # log likelihood
            posterior = prior + likelihood # calculate posterior (log)
            posteriors.append(posterior)

        if np.all(np.isneginf(posteriors)): # if all posteriors are -inf
            print(f"Warning: Test sample resulted in -inf posterior for all classes. Predicting based on prior.")
            return self._classes[np.argmax(self._priors)] # predict based on prior
        return self._classes[np.argmax(posteriors)] # return class with highest posterior

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx] # mean for the class
        var = self._var[class_idx] # variance for the class
        var = np.maximum(var, 1e-15) # safety net for variance
        numerator = np.exp(-((x - mean) ** 2) / (2 * var)) # gaussian numerator
        denominator = np.sqrt(2 * np.pi * var) # gaussian denominator
        pdf = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0) # calculate pdf, avoid division by zero
        return pdf


    def predict_proba(self, X_test):
        if self._mean is None: # check if fitted
             raise RuntimeError("Classifier must be fitted before predicting.")
        X_test = np.asarray(X_test) # convert to numpy array
        all_posteriors = [] # list for all sample posteriors
        for x in X_test: # for each sample
            posteriors = [] # posteriors for current sample
            for idx, c in enumerate(self._classes): # for each class
                if self._priors[idx] == 0: # if class had no training samples
                    posteriors.append(-np.inf) # assign very low probability
                    continue
                prior = np.log(self._priors[idx]) # log of prior
                pdf_values = self._pdf(idx, x) # calculate pdf
                safe_pdf_values = np.maximum(pdf_values, self.log_epsilon) # clip to avoid log(0)
                likelihood = np.sum(np.log(safe_pdf_values)) # log likelihood
                posterior = prior + likelihood # calculate posterior (log)
                posteriors.append(posterior)
            all_posteriors.append(posteriors) # add sample's posteriors
        return np.array(all_posteriors) # return log posteriors
