# Example Notebooks

This directory contains Jupyter notebooks demonstrating the machine learning algorithms implemented in the `jiayi_ml` package.

Each notebook shows not only how to call the package, but also how to perform a complete data science workflow:

1. Define the machine learning task.
2. State a modeling hypothesis.
3. Describe the dataset.
4. Perform exploratory data analysis.
5. Apply appropriate preprocessing.
6. Fit the model using the custom implementation.
7. Evaluate the model on held-out data or interpret unsupervised results.
8. Discuss strengths, limitations, and practical implications.

## Directory Structure

```text
examples/
├── README.md
├── supervised_learning/
│   ├── 01_linear_ridge_lasso_regression_diabetes.ipynb
│   ├── 02_logistic_regression_breast_cancer.ipynb
│   ├── 03_knn_iris.ipynb
│   ├── 04_decision_tree_random_forest_wine.ipynb
│   ├── 05_naive_bayes_breast_cancer.ipynb
│   ├── 06_perceptron_binary_classification.ipynb
│   ├── 11_linear_svm_breast_cancer.ipynb
│   └── 12_gradient_boosting_regression_diabetes.ipynb
└── unsupervised_learning/
    ├── 07_kmeans_blobs.ipynb
    ├── 08_dbscan_moons.ipynb
    ├── 09_hierarchical_clustering_wine.ipynb
    └── 10_pca_breast_cancer.ipynb
```

## Supervised Learning Notebooks

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

## Unsupervised Learning Notebooks

| Notebook | Algorithms | Dataset | Main Focus |
|---|---|---|---|
| `07_kmeans_blobs.ipynb` | K-Means | Synthetic blobs | Cluster centers, inertia, elbow method |
| `08_dbscan_moons.ipynb` | DBSCAN | Synthetic moons | Non-spherical clusters, density-based clustering, noise detection |
| `09_hierarchical_clustering_wine.ipynb` | Agglomerative Clustering | Wine dataset | Hierarchical structure, linkage methods, cluster interpretation |
| `10_pca_breast_cancer.ipynb` | PCA | Breast Cancer dataset | Dimensionality reduction, explained variance, 2D visualization |

## Dataset Choices

The notebooks use built-in datasets from `scikit-learn` and synthetic datasets generated with `sklearn.datasets`.

This choice improves reproducibility because the notebooks do not require external downloads or private datasets.

The examples are also designed to connect algorithm behavior with interpretable data science questions:

- The Diabetes dataset is used for regression, regularization, and gradient boosting.
- The Breast Cancer dataset is used for binary classification, probabilistic classification, margin-based classification, and dimensionality reduction.
- The Iris dataset is used for distance-based multiclass classification.
- The Wine dataset is used for tree-based classification and hierarchical clustering.
- Synthetic datasets are used when they provide clearer visualization of algorithm behavior.

## Notes on Evaluation

For supervised learning notebooks, models are evaluated on held-out test data using appropriate metrics.

Regression examples use:

- Mean squared error
- Root mean squared error
- Mean absolute error
- R-squared

Classification examples use:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

For unsupervised learning notebooks, results are interpreted through cluster structure, visualization, explained variance, reconstruction error, or comparison with known labels only after clustering. True labels are not used for model fitting in unsupervised examples.

## Notebook Template

Each notebook follows this general structure:

```text
1. Problem Statement
2. Modeling Hypothesis
3. Dataset Description
4. Exploratory Data Analysis
5. Preprocessing
6. Model Training
7. Evaluation or Interpretation
8. Discussion
9. Limitations
10. Conclusion
```

## Reproducibility

All notebooks are designed to run from the repository after installing the local package:

```bash
pip install -e ".[dev]"
```

The notebooks import the local `jiayi_ml` package from the repository `src/` directory.