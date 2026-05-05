from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HYPOTHESES = {
    "examples/supervised_learning/01_linear_ridge_lasso_regression_diabetes.ipynb": """## Modeling Hypothesis

Because the Diabetes dataset has a continuous outcome and standardized clinical measurements, linear models should provide a reasonable baseline for predicting disease progression.

The main hypothesis is:

> Linear regression will capture the main linear relationships between baseline measurements and disease progression, while ridge and lasso regression may improve coefficient stability through regularization.

However, because the dataset is small and disease progression is likely influenced by nonlinear and unobserved clinical factors, all three models are expected to have only moderate predictive performance. Ridge and lasso may not necessarily outperform ordinary linear regression on a single train/test split.
""",
    "examples/supervised_learning/02_logistic_regression_breast_cancer.ipynb": """## Modeling Hypothesis

Because many Breast Cancer dataset features show strong differences between malignant and benign samples, logistic regression should perform well as a linear binary classifier.

The main hypothesis is:

> A regularized logistic regression model will achieve high recall for malignant samples because the standardized tumor measurements contain strong class-separating information.

However, strong performance on this classic benchmark dataset should be interpreted cautiously. A single train/test split does not prove clinical generalization, and external validation would be needed for a real diagnostic setting.
""",
    "examples/supervised_learning/03_knn_iris.ipynb": """## Modeling Hypothesis

Because the Iris classes are well separated by petal measurements, KNN should perform well for multiclass classification.

The main hypothesis is:

> Moderate values of k will classify Iris species accurately because nearby samples in standardized feature space often belong to the same species.

Very small k values may be more sensitive to local noise, while larger k values may produce smoother decision boundaries. The effect of k may be more visible in decision boundary plots than in aggregate test metrics for this relatively easy dataset.
""",
    "examples/supervised_learning/04_decision_tree_random_forest_wine.ipynb": """## Modeling Hypothesis

Because the Wine dataset contains chemical measurements with nonlinear and threshold-like class separation, tree-based models should perform well.

The main hypothesis is:

> A random forest will generalize better than a single decision tree because averaging many trees reduces variance and makes the model less sensitive to individual splits.

A shallow tree may be more interpretable but could underfit, while an unrestricted tree may fit training data strongly but be less stable on test data.
""",
    "examples/supervised_learning/05_naive_bayes_breast_cancer.ipynb": """## Modeling Hypothesis

Because several Breast Cancer features differ strongly between malignant and benign samples, Gaussian Naive Bayes should provide a useful probabilistic baseline.

The main hypothesis is:

> Gaussian Naive Bayes will classify many samples correctly because class-conditional feature distributions contain strong signal.

However, the independence assumption is likely unrealistic because many features are correlated measurements derived from related cell nucleus characteristics. Therefore, Naive Bayes should be interpreted as a simple baseline rather than a fully realistic data-generating model.
""",
    "examples/supervised_learning/06_perceptron_binary_classification.ipynb": """## Modeling Hypothesis

Because the first synthetic dataset is approximately linearly separable, the perceptron should learn an effective linear decision boundary.

The main hypothesis is:

> The perceptron will perform well on linearly separable data but will struggle on nonlinear moon-shaped data because it can only learn a linear boundary.

This comparison is intended to show both the historical value of the perceptron and its limitation relative to more flexible nonlinear classifiers.
""",
    "examples/supervised_learning/11_linear_svm_breast_cancer.ipynb": """## Modeling Hypothesis

A linear support vector machine should perform well on this dataset because many breast cancer features show strong separation between malignant and benign samples.

The main hypothesis is:

> If the standardized features provide a roughly linearly separable representation of the two classes, then a linear SVM should achieve strong test-set classification performance by learning a maximum-margin separating hyperplane.

However, the model may still make errors because some benign and malignant samples overlap in feature space, the decision boundary is linear, and the selected regularization strength affects the margin-error trade-off.
""",
    "examples/unsupervised_learning/07_kmeans_blobs.ipynb": """## Modeling Hypothesis

Because the synthetic blob dataset was generated from compact groups, K-Means should capture much of the cluster structure.

The main hypothesis is:

> K-Means will identify meaningful distance-based groups when the clusters are compact and approximately spherical.

However, if two generated groups overlap or lie close together, K-Means may merge or split them imperfectly. The elbow method may therefore suggest fewer effective clusters than the true number of generating centers.
""",
    "examples/unsupervised_learning/08_dbscan_moons.ipynb": """## Modeling Hypothesis

Because the moon-shaped dataset contains non-spherical clusters, DBSCAN should be more appropriate than K-Means.

The main hypothesis is:

> DBSCAN will recover the two curved moon-shaped groups by using density connectivity rather than centroid distance.

However, the result should be sensitive to `eps` and `min_samples`. If `eps` is too small or `min_samples` is too strict, the algorithm may fragment the data into many clusters and label many points as noise.
""",
    "examples/unsupervised_learning/09_hierarchical_clustering_wine.ipynb": """## Modeling Hypothesis

Because the Wine dataset contains chemical measurements from known cultivars, hierarchical clustering may recover some class-related structure without using labels.

The main hypothesis is:

> Linkage choice will strongly affect the clustering result because different linkage rules define cluster distance differently.

Single linkage may suffer from chaining, average linkage may create imbalanced clusters, and complete linkage may produce more compact and balanced clusters for this standardized dataset.
""",
    "examples/unsupervised_learning/10_pca_breast_cancer.ipynb": """## Modeling Hypothesis

Because the Breast Cancer dataset contains many correlated numeric measurements, PCA should be able to summarize much of the feature variation using fewer components.

The main hypothesis is:

> The first few principal components will capture a substantial portion of total variance, and a two-dimensional PCA projection may reveal structure related to malignant and benign samples.

However, PCA is unsupervised and does not use diagnosis labels when fitting. Therefore, visible class separation in a PCA plot should be interpreted as exploratory structure rather than supervised classification performance.
""",
}


def has_hypothesis_cell(notebook):
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and "## Modeling Hypothesis" in cell.source:
            return True
    return False


def find_insert_index(notebook):
    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and "## 1. Problem Statement" in cell.source:
            return idx + 1
    return 1


def main():
    updated = []

    for relative_path, hypothesis_text in HYPOTHESES.items():
        notebook_path = PROJECT_ROOT / relative_path

        if not notebook_path.exists():
            print(f"Missing notebook, skipped: {relative_path}")
            continue

        notebook = nbf.read(notebook_path, as_version=4)

        if has_hypothesis_cell(notebook):
            print(f"Already has hypothesis, skipped: {relative_path}")
            continue

        insert_index = find_insert_index(notebook)
        notebook.cells.insert(insert_index, nbf.v4.new_markdown_cell(hypothesis_text))

        nbf.write(notebook, notebook_path)
        updated.append(relative_path)
        print(f"Added hypothesis: {relative_path}")

    print()
    print(f"Updated {len(updated)} notebooks.")


if __name__ == "__main__":
    main()