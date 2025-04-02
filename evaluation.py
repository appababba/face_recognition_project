import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def calculate_accuracy(y_true, y_pred):
    """Calculates classification accuracy."""
    return accuracy_score(y_true, y_pred)

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plots a confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show() # Or save figure

def plot_roc_curves_ovr(y_true, y_scores, class_labels):
    """
    Plots ROC curves for each class using One-vs-Rest (OvR) strategy.

    Args:
        y_true (array): True labels (integers).
        y_scores (array): Target scores. Can either be probability estimates
                          of the positive class, confidence values, or binary decisions.
                          Shape (n_samples, n_classes).
        class_labels (list/array): Unique class labels corresponding to columns in y_scores.
    """
    n_classes = len(class_labels)
    # Binarize the true labels for OvR
    y_true_binarized = label_binarize(y_true, classes=class_labels)

    # Check if y_true_binarized is 1D (only 2 classes)
    if n_classes == 2 and y_true_binarized.ndim == 1:
        y_true_binarized = np.column_stack((1 - y_true_binarized, y_true_binarized))
        # Ensure y_scores also handles the 2-class case appropriately
        if y_scores.ndim == 1:
             y_scores = np.column_stack((1 - y_scores, y_scores)) # Assuming scores are for positive class
        elif y_scores.shape[1] == 1:
             y_scores = np.column_stack((1 - y_scores[:,0], y_scores[:,0]))


    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        # Ensure we are accessing the correct column for scores corresponding to class i
        # Naive Bayes might output log probabilities, kNN simple probabilities
        # Ensure scores are probability of the positive class for OvR
        # This might need adjustment based on how predict_proba is implemented
        scores_for_class_i = y_scores[:, i]

        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], scores_for_class_i)
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC curve of class {class_labels[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Dashed diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) - One-vs-Rest')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show() # Or save figure

# --- FMR/FNMR Calculation (More complex for multi-class ID) ---
# This requires defining "genuine" vs "impostor" attempts based on scores.
# It's often easier to calculate from the ROC curve points (FPR = FMR, FNR = 1-TPR = FNMR)
# Example: Find TPR at a specific desired FPR (FMR)
def find_tpr_at_fpr(fpr_array, tpr_array, desired_fpr):
     """Finds the TPR corresponding to the closest FPR <= desired_fpr."""
     if desired_fpr < fpr_array[0]: return 0.0 # Handle edge case
     # Find the index of the FPR value just below or equal to the desired FPR
     idx = np.where(fpr_array <= desired_fpr)[0][-1]
     return tpr_array[idx]

# You would call this inside a loop for each class's ROC curve points from plot_roc_curves_ovr
# print(f"Class {label}: At FMR (FPR) <= {desired_fpr:.4f}, TPR is {tpr:.4f} (FNMR = {1-tpr:.4f})")