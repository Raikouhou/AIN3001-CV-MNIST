# MNIST Digit Classification with SVM & Hyperparameter Optimization

A machine learning project that trains a **Support Vector Machine (SVM)** classifier on the MNIST handwritten digit dataset, using **cross-validation**, **RandomizedSearchCV**, and **GridSearchCV** to systematically find the best hyperparameters and maximize classification accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [1. Data Loading & Preprocessing](#1-data-loading--preprocessing)
  - [2. Baseline SVM](#2-baseline-svm)
  - [3. Cross-Validation](#3-cross-validation)
  - [4. RandomizedSearchCV](#4-randomizedsearchcv)
  - [5. GridSearchCV](#5-gridsearchcv)
  - [6. Final Model & Evaluation](#6-final-model--evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Next Steps](#next-steps)

---

## Overview

This project explores how to build a reliable digit classifier using SVMs — a classical and powerful ML algorithm — on the iconic MNIST dataset. Rather than just training a single model, the focus is on the **full hyperparameter optimization pipeline**: starting from a default baseline, then progressively narrowing down the best configuration using randomized and exhaustive search strategies paired with cross-validation.

The final optimized model achieves **~97% accuracy** on the test set despite being trained on only 15% of the full dataset.

---

## Dataset

**MNIST (Modified National Institute of Standards and Technology)**

- 70,000 grayscale images of handwritten digits (0–9)
- Each image is **28×28 pixels**, flattened to a 784-dimensional feature vector
- Pixel values range from 0 (black) to 255 (white)
- Loaded directly via `sklearn.datasets.fetch_openml("mnist_784", version=1)`

---

## Project Structure

```
CV_with_MNIST_Dataset.ipynb   # Main notebook
README.md                     # This file
```

---

## Methodology

### 1. Data Loading & Preprocessing

The dataset is fetched from OpenML and split into features `X` (pixel arrays) and labels `y` (digit classes). The target labels are cast from string objects to `int8` for numerical compatibility.

```python
data = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = data.data, data.target
y = y.astype(np.int8)
```

A 5×5 grid of sample images is plotted to visually verify the data looks correct before training.

**Train/Test Split:**  
Due to the computational cost of SVM training on large datasets (especially with `rbf` and `poly` kernels), a **15% training / 85% test** split is used to keep training time feasible. The training set is checked for class balance to ensure no digit is under-represented.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.85, random_state=42)
```

---

### 2. Baseline SVM

A default `SVC()` is trained with no hyperparameter tuning to establish a performance baseline.

```python
svc = SVC()
svc.fit(X_train, y_train)
```

This default model uses an RBF kernel with `C=1.0` and `gamma='scale'`.

---

### 3. Cross-Validation

To get a reliable performance estimate for the baseline model, **5-fold cross-validation** is applied on the training set. This splits the training data into 5 folds, trains on 4 and validates on 1, rotating through all combinations — giving a much more honest picture of model generalization than a single train/test split.

```python
cv_score = cross_val_score(svc, X_train, y_train, cv=5)
print("Average score:", round(cv_score.mean(), 4) * 100, "%")
```

---

### 4. RandomizedSearchCV

Instead of trying every possible combination of hyperparameters (which would be extremely slow), `RandomizedSearchCV` samples a **random subset** of the parameter space. This is run three times with increasing numbers of iterations (`n_iter = 3`, `10`, `15`) to observe how more sampling affects the result.

**Parameter search space:**

| Parameter | Values Searched         |
|-----------|-------------------------|
| `C`       | 0.1, 1, 10, 100         |
| `gamma`   | `'scale'`, `'auto'`     |
| `kernel`  | `'linear'`, `'rbf'`, `'poly'` |

Each search uses **5-fold CV** internally to score each sampled configuration.

```python
param_dist = {"C": [0.1, 1, 10, 100], "gamma": ['scale', 'auto'], "kernel": ['linear', 'rbf', 'poly']}

randomized_search = RandomizedSearchCV(SVC(), param_dist, n_iter=3, cv=5, random_state=42, n_jobs=-1)
```

All three runs converge on the same best configuration: **C=10, kernel=rbf, gamma=scale**.

---

### 5. GridSearchCV

With `C=10` established as the best regularization value from RandomizedSearchCV, a **GridSearchCV** exhaustively tests all combinations of `kernel` and `gamma` while fixing `C=10`. This is more thorough than random sampling within the narrowed space.

```python
param_grid_gsv = {"C": [10], "gamma": ['scale', 'auto'], "kernel": ['linear', 'rbf', 'poly']}

grid_search = GridSearchCV(SVC(), param_grid_gsv, cv=5, n_jobs=-1)
```

GridSearchCV confirms the same optimal configuration, giving a **cross-validation score of ~96.41%**.

---

### 6. Final Model & Evaluation

The best configuration is used to train a final model on the full training set, then evaluated on the held-out test set.

```python
best_model = SVC(C=10, kernel="rbf", gamma="scale")
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
```

Evaluation includes:
- **Confusion Matrix** — a heatmap showing where predictions are correct vs. where digits are confused for each other
- **Classification Report** — per-class precision, recall, and F1 scores

---

## Results

| Model | Best Params | CV Score |
|---|---|---|
| Baseline SVC | `C=1, kernel=rbf, gamma=scale` | ~95% |
| RandomizedSearchCV (n_iter=3) | `C=10, kernel=rbf, gamma=scale` | 96.41% |
| RandomizedSearchCV (n_iter=10) | `C=10, kernel=rbf, gamma=scale` | 96.41% |
| RandomizedSearchCV (n_iter=15) | `C=10, kernel=rbf, gamma=scale` | 96.41% |
| GridSearchCV | `C=10, kernel=rbf, gamma=scale` | 96.41% |

**Final test set performance (C=10, rbf, gamma=scale):**
- Accuracy: **~97%**
- Precision, Recall, F1: **>95%** across all digit classes
- Trained on only **15%** of the full 70,000-image dataset

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
tabulate
```

Install all dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn tabulate
```

---

## How to Run

1. Clone the repository and open the notebook:

```bash
git clone https://github.com/your-username/mnist-svm-classifier.git
cd mnist-svm-classifier
jupyter notebook CV_with_MNIST_Dataset.ipynb
```

2. Run all cells top to bottom. The MNIST dataset will be downloaded automatically via `fetch_openml` on first run (requires internet connection).

> **Note:** RandomizedSearchCV and GridSearchCV cells can take several minutes to complete depending on your hardware. `n_jobs=-1` is set to use all available CPU cores.

---

## Next Steps

- Train on a larger portion of the dataset (e.g., 70/30 split) to further improve accuracy
- Investigate the cause of low minimum cross-validation scores in some folds
- Experiment with other classifiers (Random Forest, k-NN, Logistic Regression) for comparison
- Train a **Convolutional Neural Network (CNN)** — the standard architecture for image classification tasks — and compare against the SVM baseline
- Use **stratified splitting** to guarantee balanced class distribution across folds

---

*Author: Yavuz Selim Vurgun*
