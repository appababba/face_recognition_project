# CSci 158 Face Recognition Project - Spring 2025

This is our group project for CSci 158, where we built a face recognition system for identifying people.

**Team:** James Lin, Robert Cameron

## Goal

The main goal was to implement a couple of classification algorithms **from scratch** (as required by the assignment!) and see how well they work for face identification on a standard dataset. We chose to implement:

1.  **K-Nearest Neighbors (KNN)**
2.  **Gaussian Naive Bayes (GNB)**

We compare their performance using metrics like accuracy, confusion matrices, and ROC curves.

## What The Code Does

The main script (`main.py`) orchestrates the whole process:

1.  **Loads Data:** Reads face images and labels from a specified dataset directory (`data_loader.py`). Assumes images are organized into subfolders for each person.
2.  **Preprocesses Images:** Resizes images to a standard size, converts them to grayscale, and flattens them into feature vectors (`preprocessing.py`).
3.  **Extracts Features:** Currently set up to use **PCA** for dimensionality reduction (using scikit-learn) or just use the **raw** (scaled) pixel values.
4.  **Trains Classifiers:** Trains the KNN and GNB classifiers implemented from scratch in `classifiers.py`.
5.  **Evaluates:** Tests the classifiers on a held-out test set, calculates accuracy, and generates confusion matrix and ROC curve plots (`evaluation.py`), saving them as PNG files.

## Setup Instructions

1.  **Python:** You'll need Python 3 (we developed using Python 3.13, but slightly older versions should work).
2.  **Virtual Environment (Recommended):** It's best practice to use a virtual environment to keep dependencies clean.
    * Create it: `python3 -m venv .venv`
    * Activate it:
        * macOS/Linux: `source .venv/bin/activate`
        * Windows: `.venv\Scripts\activate`
3.  **Install Libraries:** Once the environment is active, install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```
    (This installs `numpy`, `opencv-python`, `scikit-learn`, `matplotlib`, `seaborn`).

## Getting the Data

* The code expects image data in the `data/` directory.
* We primarily tested with the **AT&T Database of Faces (ORL Database)**. You can find this online at [https://www.kaggle.com/datasets/kasikrit/att-database-of-faces?resource=download].
* Download and unzip the dataset.
* Create a folder inside `data/` (e.g., `data/att_faces/` or `data/archive/`).
* Place the subject folders (`s1`, `s2`, `s3`, etc.) directly inside the folder you created. The structure should look like `data/your_dataset_name/subject_folder/image_file.pgm`.

## How to Run

1.  **Activate Environment:** Make sure your virtual environment is active (you should see `(.venv)` in your terminal prompt).
2.  **Configure `main.py`:**
    * Open `main.py`.
    * **Set `DATASET_PATH`** to the correct path relative to `main.py` where your subject folders (`s1`, `s2`, etc.) are located (e.g., `DATASET_PATH = './data/att_faces/'`).
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

* We implemented the classifiers from scratch as required. Libraries like `scikit-learn` were used for helpful things like PCA, train/test split, scaling, and some evaluation metrics (like generating the confusion matrix values and ROC points).
* Getting the plots to work without hanging was a bit tricky! We ended up modifying the evaluation code to save plots directly to files (`plt.savefig`) instead of trying to display them interactively (`plt.show`).
* Also had some fun debugging the Gaussian Naive Bayes `log(0)` errors - needed to clip tiny probability values before taking the log to avoid getting `-infinity`!

---
