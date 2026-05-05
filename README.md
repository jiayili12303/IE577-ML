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
│   ├── supervised_learning/
│   └── unsupervised_learning/
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

The supervised learning module currently includes:

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

### Unsupervised Learning

The unsupervised learning module currently includes:

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

The `examples/` directory will contain Jupyter notebooks demonstrating the algorithms on real or synthetic datasets.

Planned supervised learning notebooks:

- Linear / Ridge / Lasso Regression on the Diabetes dataset
- Logistic Regression on the Breast Cancer dataset
- KNN on the Iris dataset
- Decision Tree and Random Forest models on the Wine dataset
- Gaussian Naive Bayes on the Breast Cancer dataset
- Perceptron on a binary classification dataset

Planned unsupervised learning notebooks:

- K-Means clustering on synthetic blob data
- DBSCAN clustering on synthetic moon-shaped data
- Hierarchical clustering on the Wine dataset
- PCA on the Breast Cancer dataset

Each notebook is designed to include:

1. Problem statement
2. Dataset description
3. Exploratory data analysis
4. Preprocessing
5. Model training
6. Test-set evaluation or cluster interpretation
7. Discussion of results
8. Limitations

## Development Practices

This repository follows several software engineering practices:

- Modular package structure under `src/`
- Reusable model classes with `fit`, `predict`, `fit_predict`, `transform`, or `score` methods
- NumPy-based algorithm implementations
- Docstrings for functions and classes
- Unit tests using `pytest`
- Continuous integration using GitHub Actions
- Project configuration through `pyproject.toml`

## Author

Jiayi Li  
Rice University  
CMOR 438 / INDE 577: Data Science and Machine Learning
