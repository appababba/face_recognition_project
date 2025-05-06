import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def calculate_accuracy(y_true, y_pred):
    # calculates classification accuracy
    return accuracy_score(y_true, y_pred)

def plot_confusion_matrix(y_true, y_pred, class_names, name=""):
    #plots a confusion matrix using seaborn and saves it
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true))) # make sure labels match class_names order
    # make sure class_names corresponds correctly if not all classes appear in y_true/y_pred
    tick_labels = class_names
    if len(class_names) != cm.shape[0]:
         print(f"Number of class names ({len(class_names)}) doesn't match confusion matrix size ({cm.shape[0]}). Using numeric labels.")
         tick_labels = sorted(np.unique(y_true))

    plt.figure(figsize=(12, 10)) # adjusted size slightly
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout() # adjust layout
    plt.savefig(f'confusion_matrix_{name}.png', dpi=150) # save the plot, increase dpi
    plt.close() # close the plot object

def plot_roc_curves_ovr(y_true, y_scores, class_labels, name=""):
    """
    plots ROC curves for each class using One-vs-Rest (OvR) strategy and saves it.
    """
    n_classes = len(class_labels)
    # binarize the true labels for OvR using the sorted class labels
    y_true_binarized = label_binarize(y_true, classes=class_labels)

    # check for edge case
    if y_true_binarized.shape[1] == 1 and n_classes > 1:
        print(f"Only one class present in y_true for ROC calculation.")
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.text(0.5, 0.5, 'ROC undefined (only one class in y_true)', ha='center', va='center', fontsize=12)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curve (Undefined) - {name}')
        plt.savefig(f'roc_curve_{name}_undefined.png')
        plt.close()
        return 
        
    if np.max(y_scores) <= 0 and np.isneginf(np.min(y_scores)):
        print("Detected log-probabilities, converting to probabilities for ROC.")

        max_log_scores = np.max(y_scores, axis=1, keepdims=True)
        exp_scores = np.exp(y_scores - max_log_scores)
        # normalize to get probabilities
        sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
        y_scores = np.divide(exp_scores, sum_exp_scores, out=np.full_like(exp_scores, 1.0/n_classes), where=sum_exp_scores!=0)


    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        class_idx_label = class_labels[i] # the actual label value
        # check if this class exists in y_true to avoid errors with binarized shape
        if class_idx_label not in np.unique(y_true):
             print(f"Skipping ROC for class {class_idx_label} - not present in y_true.")
             continue
        # this handles cases where classes might be missing from y_true after split
        try:
            true_column_idx = list(class_labels).index(class_idx_label)
            if true_column_idx >= y_true_binarized.shape[1]:
                 print(f"Index mismatch for class {class_idx_label}. Re-checking binarization.")
                 present_classes = sorted(np.unique(y_true))
                 y_true_binarized_local = label_binarize(y_true, classes=present_classes)
                 true_column_idx = list(present_classes).index(class_idx_label)
                 target_y_true = y_true_binarized_local[:, true_column_idx]
            else:
                 target_y_true = y_true_binarized[:, true_column_idx]

            target_y_score = y_scores[:, i] # scores column from sorted class_labels

            fpr[i], tpr[i], _ = roc_curve(target_y_true, target_y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2,
                     label=f'ROC curve of class {class_labels[i]} (area = {roc_auc[i]:0.2f})')
        except ValueError as e:
            print(f"Could not calculate ROC for class {class_labels[i]}: {e}. Skipping.")
            continue
        except IndexError as e:
             print(f"IndexError calculating ROC for class {class_labels[i]}: {e}. Skipping. "
                   f"(Num classes: {n_classes}, Binarized shape: {y_true_binarized.shape}, Scores shape: {y_scores.shape})")
             continue


    plt.plot([0, 1], [0, 1], 'k--', lw=2) # dashed diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Receiver Operating Characteristic (ROC) - One-vs-Rest - {name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout() # adjust layout
    plt.savefig(f'roc_curve_{name}.png', dpi=150) # save the plot
    plt.close() # close the plot object
