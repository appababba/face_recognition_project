# CSCI 158 Face Recognition Project - Spring 2025

This is our group project for CSci 158, where we built a face recognition system for identifying people.

**Team:** James Lin, Robert Cameron

## Goal

The main goal was to implement a couple of classification algorithms **from scratch** and see how well they work for face identification on a standard dataset. We chose to implement:

1.  **K-Nearest Neighbors (KNN)**
2.  **Gaussian Naive Bayes (GNB)**

We compare their performance using metrics like accuracy, confusion matrices, and ROC curves.

## What The Code Does

The main script (`main.py`):

1.  **Loads Data:** Reads face images and labels from (`data_loader.py`). Images are organized into subfolders for each person.
2.  **Preprocesses Images:** Resizes images to a standard size, converts them to grayscale, and flattens them into feature vectors.
3.  **Extracts Features:** Currently set up to use **PCA** for dimensionality reduction or just use the scaled (scaled) pixel values.
4.  **Trains Classifiers:** Trains the KNN and GNB classifiers implemented in `classifiers.py`.
5.  **Evaluates:** Tests the classifiers on a held-out test set, calculates accuracy, and generates confusion matrix and ROC curve plots (`evaluation.py`), and saves them as PNG files.

## Setup Instructions

1.  **Virtual Environment:** You should use a virtual environment. 
    * Creation: `python3 -m venv .venv`
    * Activation:
        * macOS/Linux: `source .venv/bin/activate`
        * Windows: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force`,
`..\.venv\Scripts\activate.ps1`
2.  **Install Libraries:** Once the environment is active, install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```
    This installs `numpy`, `opencv-python`, `scikit-learn`, `matplotlib`, `seaborn`.

## Getting the Data

* The code expects image data in the `data/` directory.
* We tested with the **AT&T Database of Faces (ORL Database)**. You can find this online at [https://www.kaggle.com/datasets/kasikrit/att-database-of-faces?resource=download].
* Download and unzip the dataset.
* Create a folder inside `data/` 
* Place the subject folders  directly inside the folder. The path will look like `data/dataset_name/subject_folder/image_file.pgm`.

## How to Run

1.  **Activate Environment:** Make sure the virtual environment is active (the terminal will say `(.venv)`)
2.  **Configure `main.py`:**
    * Open `main.py`.
    * **Set `DATASET_PATH`** to the correct path relative to `main.py` where (`s1`, `s2`...) are.
    * You can also change `FEATURE_EXTRACTION_METHOD` (set to `'pca'` or `'raw'`), `N_PCA_COMPONENTS` (if using PCA), and `KNN_K` for experiments.
3.  **Run from Terminal:**
    ```bash
    python main.py
    ```

## Output

The script will print progress messages and the final accuracy scores for KNN and GaussianNB to the terminal.

It will also save the following plots as PNG files in the main project directory:

* `confusion_matrix_KNN (Scratch).png`
* `confusion_matrix_GaussianNB (Scratch).png`
* `roc_curve_KNN (Scratch).png`
* `roc_curve_GaussianNB (Scratch).png`

## Notes / Challenges

* We implemented the classifiers from scratch, and used libraries like `scikit-learn` for helpful things like PCA, train/test split, scaling, and some evaluation metrics (like generating the confusion matrix values and ROC points).

---
