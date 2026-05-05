# Example Notebooks

This directory contains Jupyter notebooks demonstrating the machine learning algorithms implemented in the `jiayi_ml` package.

Each notebook is designed to show not only how to call the package, but also how to perform a complete data science workflow:

1. Define the machine learning task.
2. Describe the dataset.
3. Perform exploratory data analysis.
4. Apply appropriate preprocessing.
5. Fit the model using the custom implementation.
6. Evaluate the model on held-out data or interpret clustering results.
7. Discuss strengths, limitations, and practical implications.

## Directory Structure

```text
examples/
├── README.md
├── supervised_learning/
└── unsupervised_learning/
```

## Planned Supervised Learning Notebooks

| Notebook | Algorithms | Dataset | Main Focus |
|---|---|---|---|
| `01_linear_ridge_lasso_regression_diabetes.ipynb` | Linear Regression, Ridge Regression, Lasso Regression | Diabetes dataset | Regression performance, regularization, coefficient shrinkage |
| `02_logistic_regression_breast_cancer.ipynb` | Logistic Regression | Breast Cancer dataset | Binary classification, confusion matrix, recall, false negatives |
| `03_knn_iris.ipynb` | KNN Classifier, KNN Regressor | Iris dataset and simple regression data | Effect of k, distance-based learning, decision boundaries |
| `04_decision_tree_random_forest_wine.ipynb` | Decision Tree, Random Forest | Wine dataset | Tree-based classification, overfitting, ensemble stability, feature importance |
| `05_naive_bayes_breast_cancer.ipynb` | Gaussian Naive Bayes | Breast Cancer dataset | Probabilistic classification, independence assumptions |
| `06_perceptron_binary_classification.ipynb` | Perceptron | Synthetic binary classification dataset | Linear separability, mistake-driven learning |

## Planned Unsupervised Learning Notebooks

| Notebook | Algorithms | Dataset | Main Focus |
|---|---|---|---|
| `07_kmeans_blobs.ipynb` | K-Means | Synthetic blobs | Cluster centers, inertia, elbow method |
| `08_dbscan_moons.ipynb` | DBSCAN | Synthetic moons | Non-spherical clusters, density-based clustering, noise detection |
| `09_hierarchical_clustering_wine.ipynb` | Agglomerative Clustering | Wine dataset | Hierarchical structure, linkage methods, cluster interpretation |
| `10_pca_breast_cancer.ipynb` | PCA | Breast Cancer dataset | Dimensionality reduction, explained variance, 2D visualization |

## Dataset Choices

The notebooks use built-in datasets from `scikit-learn` and synthetic datasets generated with `sklearn.datasets`.

This choice improves reproducibility because the notebooks do not require external downloads or private datasets.

The examples are also designed to connect algorithm behavior with interpretable data science questions. For example:

- The Diabetes dataset is used for regression and regularization.
- The Breast Cancer dataset is used for clinically motivated binary classification and dimensionality reduction.
- The Wine dataset is used for tree-based classification and hierarchical clustering.
- Synthetic datasets are used when they provide clearer visualization of algorithm behavior.

## Notes on Evaluation

For supervised learning notebooks, models are evaluated on held-out test data using appropriate metrics.

For unsupervised learning notebooks, results are interpreted through cluster structure, visualization, or comparison with known labels only after clustering. True labels are not used for model fitting in unsupervised examples.

## Notebook Template

Each notebook should follow this structure:

```text
1. Problem Statement
2. Dataset Description
3. Exploratory Data Analysis
4. Preprocessing
5. Model Training
6. Evaluation or Interpretation
7. Discussion
8. Limitations
```