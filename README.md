# IE577-ML: Machine Learning Algorithms from Scratch

[![Run Unit Tests](https://github.com/jiayili12303/IE577-ML/actions/workflows/tests.yml/badge.svg)](https://github.com/jiayili12303/IE577-ML/actions/workflows/tests.yml)

This repository contains a NumPy-based machine learning package implemented from scratch for **CMOR 438 / INDE 577: Data Science and Machine Learning**.

The project focuses on both algorithmic implementation and reproducible data science practice. It includes supervised learning algorithms, unsupervised learning algorithms, preprocessing utilities, evaluation metrics, unit tests, GitHub Actions continuous integration, and example notebooks.

## Project Goals

The goals of this project are to:

1. Implement core machine learning algorithms from scratch using NumPy.
2. Build a reusable Python package with a clean project structure.
3. Provide unit tests for correctness and reliability.
4. Use GitHub Actions to automatically run tests on every push.
5. Demonstrate each algorithm through Jupyter notebooks with data exploration, preprocessing, model fitting, evaluation, and interpretation.

## Repository Structure

```text
IE577-ML/
├── .github/
│   └── workflows/
│       └── tests.yml
├── examples/
│   ├── README.md
│   ├── supervised_learning/
│   │   ├── 01_linear_ridge_lasso_regression_diabetes.ipynb
│   │   ├── 02_logistic_regression_breast_cancer.ipynb
│   │   ├── 03_knn_iris.ipynb
│   │   ├── 04_decision_tree_random_forest_wine.ipynb
│   │   ├── 05_naive_bayes_breast_cancer.ipynb
│   │   ├── 06_perceptron_binary_classification.ipynb
│   │   ├── 11_linear_svm_breast_cancer.ipynb
│   │   └── 12_gradient_boosting_regression_diabetes.ipynb
│   └── unsupervised_learning/
│       ├── 07_kmeans_blobs.ipynb
│       ├── 08_dbscan_moons.ipynb
│       ├── 09_hierarchical_clustering_wine.ipynb
│       └── 10_pca_breast_cancer.ipynb
├── scripts/
├── src/
│   └── jiayi_ml/
│       ├── preprocessing/
│       ├── metrics/
│       ├── supervised/
│       └── unsupervised/
├── tests/
├── pyproject.toml
├── requirements.txt
├── pytest.ini
└── README.md
```

## Package Overview

The package is organized into four main modules:

```python
jiayi_ml.preprocessing
jiayi_ml.metrics
jiayi_ml.supervised
jiayi_ml.unsupervised
```

### Preprocessing

The preprocessing module includes:

- `StandardScaler`
- `MinMaxScaler`

These classes follow a scikit-learn-like API:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Metrics

The metrics module includes regression and classification metrics.

Regression metrics:

- `mean_squared_error`
- `root_mean_squared_error`
- `mean_absolute_error`
- `r2_score`

Classification metrics:

- `accuracy_score`
- `confusion_matrix`
- `precision_score`
- `recall_score`
- `f1_score`

## Implemented Algorithms

### Supervised Learning

The supervised learning module includes:

| Algorithm | Class |
|---|---|
| Linear Regression | `LinearRegression` |
| Ridge Regression | `RidgeRegression` |
| Lasso Regression | `LassoRegression` |
| Logistic Regression | `LogisticRegression` |
| K-Nearest Neighbors Classifier | `KNNClassifier` |
| K-Nearest Neighbors Regressor | `KNNRegressor` |
| Gaussian Naive Bayes | `GaussianNaiveBayes` |
| Decision Tree Classifier | `DecisionTreeClassifier` |
| Decision Tree Regressor | `DecisionTreeRegressor` |
| Random Forest Classifier | `RandomForestClassifier` |
| Random Forest Regressor | `RandomForestRegressor` |
| Perceptron | `Perceptron` |
| Linear Support Vector Machine | `LinearSVM` |
| Gradient Boosting Regressor | `GradientBoostingRegressor` |

### Unsupervised Learning

The unsupervised learning module includes:

| Algorithm | Class |
|---|---|
| K-Means Clustering | `KMeans` |
| Principal Component Analysis | `PCA` |
| DBSCAN | `DBSCAN` |
| Agglomerative Hierarchical Clustering | `AgglomerativeClustering` |

## Installation

Clone the repository:

```bash
git clone https://github.com/jiayili12303/IE577-ML.git
cd IE577-ML
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -e .
```

For development, install optional development dependencies:

```bash
pip install -e ".[dev]"
```

## Running Tests

This project uses `pytest` for unit testing.

Run all tests locally with:

```bash
python -m pytest
```

The repository also uses GitHub Actions to automatically run tests on every push and pull request.

## Example Usage

### Linear Regression

```python
import numpy as np
from jiayi_ml.supervised import LinearRegression

X = np.array([[0], [1], [2], [3]])
y = np.array([1, 3, 5, 7])

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
print(predictions)
```

### Logistic Regression

```python
import numpy as np
from jiayi_ml.supervised import LogisticRegression

X = np.array([[-2], [-1], [1], [2]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression(learning_rate=0.1, max_iter=1000)
model.fit(X, y)

predictions = model.predict(X)
probabilities = model.predict_proba(X)

print(predictions)
print(probabilities)
```

### Linear SVM

```python
import numpy as np
from jiayi_ml.supervised import LinearSVM

X = np.array([[-2], [-1], [1], [2]])
y = np.array([0, 0, 1, 1])

model = LinearSVM(learning_rate=0.05, max_iter=5000, alpha=0.01)
model.fit(X, y)

predictions = model.predict(X)
decision_scores = model.decision_function(X)

print(predictions)
print(decision_scores)
```

### Gradient Boosting Regressor

```python
import numpy as np
from jiayi_ml.supervised import GradientBoostingRegressor

X = np.array([[0], [1], [2], [3], [4]])
y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

model = GradientBoostingRegressor(
    n_estimators=20,
    learning_rate=0.1,
    max_depth=2,
)
model.fit(X, y)

predictions = model.predict(X)
print(predictions)
```

### K-Means

```python
import numpy as np
from jiayi_ml.unsupervised import KMeans

X = np.array([
    [0.0, 0.0],
    [0.1, 0.0],
    [10.0, 10.0],
    [10.1, 10.0],
])

model = KMeans(n_clusters=2, random_state=438)
labels = model.fit_predict(X)

print(labels)
```

## Example Notebooks

The `examples/` directory contains Jupyter notebooks demonstrating the algorithms on real and synthetic datasets.

### Supervised Learning Notebooks

| Notebook | Algorithms | Dataset | Main Focus |
|---|---|---|---|
| `01_linear_ridge_lasso_regression_diabetes.ipynb` | Linear Regression, Ridge Regression, Lasso Regression | Diabetes dataset | Regression performance, regularization, coefficient shrinkage |
| `02_logistic_regression_breast_cancer.ipynb` | Logistic Regression | Breast Cancer dataset | Binary classification, confusion matrix, recall, false negatives |
| `03_knn_iris.ipynb` | KNN Classifier, KNN Regressor | Iris dataset and synthetic regression data | Effect of k, distance-based learning, decision boundaries |
| `04_decision_tree_random_forest_wine.ipynb` | Decision Tree, Random Forest | Wine dataset | Tree-based classification, overfitting, ensemble stability, feature importance |
| `05_naive_bayes_breast_cancer.ipynb` | Gaussian Naive Bayes | Breast Cancer dataset | Probabilistic classification, independence assumptions |
| `06_perceptron_binary_classification.ipynb` | Perceptron | Synthetic binary classification datasets | Linear separability, mistake-driven learning, nonlinear limitations |
| `11_linear_svm_breast_cancer.ipynb` | Linear SVM | Breast Cancer dataset | Margin-based binary classification, hinge loss, regularization, decision scores |
| `12_gradient_boosting_regression_diabetes.ipynb` | Gradient Boosting Regressor | Diabetes dataset | Additive ensemble regression, residual fitting, learning rate and estimator sensitivity |

### Unsupervised Learning Notebooks

| Notebook | Algorithms | Dataset | Main Focus |
|---|---|---|---|
| `07_kmeans_blobs.ipynb` | K-Means | Synthetic blobs | Cluster centers, inertia, elbow method |
| `08_dbscan_moons.ipynb` | DBSCAN | Synthetic moons | Non-spherical clusters, density-based clustering, noise detection |
| `09_hierarchical_clustering_wine.ipynb` | Agglomerative Clustering | Wine dataset | Hierarchical structure, linkage methods, cluster interpretation |
| `10_pca_breast_cancer.ipynb` | PCA | Breast Cancer dataset | Dimensionality reduction, explained variance, 2D visualization |

Each notebook includes:

1. Problem statement
2. Modeling hypothesis
3. Dataset description
4. Exploratory data analysis
5. Preprocessing
6. Model training
7. Evaluation or interpretation
8. Discussion of results
9. Limitations
10. Conclusion

## Dataset Choices

The notebooks use built-in datasets from `scikit-learn` and synthetic datasets generated with `sklearn.datasets`.

This choice improves reproducibility because the notebooks do not require external downloads or private datasets.

The examples are designed to connect algorithm behavior with interpretable data science questions:

- The Diabetes dataset is used for regression, regularization, and gradient boosting.
- The Breast Cancer dataset is used for binary classification, probabilistic classification, margin-based classification, and dimensionality reduction.
- The Iris dataset is used for distance-based multiclass classification.
- The Wine dataset is used for tree-based classification and hierarchical clustering.
- Synthetic datasets are used when they provide clearer visualization of algorithm behavior.

## Development Practices

This repository follows several software engineering practices:

- Modular package structure under `src/`
- Reusable model classes with `fit`, `predict`, `fit_predict`, `transform`, or `score` methods
- NumPy-based algorithm implementations
- Docstrings for functions and classes
- Unit tests using `pytest`
- Continuous integration using GitHub Actions
- Project configuration through `pyproject.toml`

## Testing and Continuous Integration

The test suite covers:

- Package imports
- Preprocessing utilities
- Regression metrics
- Classification metrics
- Supervised learning algorithms
- Unsupervised learning algorithms
- Error handling for invalid inputs
- Model behavior on simple known examples

GitHub Actions automatically runs the unit tests on every push and pull request using Python 3.11 and Python 3.12.

## Notes

This package is built for educational purposes. It prioritizes clarity, readability, and algorithmic understanding over production-level optimization.

The implementations are intended to demonstrate the core ideas behind machine learning algorithms rather than replace mature libraries such as scikit-learn.

## Author

Jiayi Li  
Rice University  
CMOR 438 / INDE 577: Data Science and Machine Learning