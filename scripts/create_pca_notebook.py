from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "examples"
    / "unsupervised_learning"
    / "10_pca_breast_cancer.ipynb"
)

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Principal Component Analysis on the Breast Cancer Dataset

This notebook demonstrates the `PCA` algorithm implemented in the `jiayi_ml` package.

The goal is to reduce a high-dimensional dataset to a lower-dimensional representation while preserving as much variance as possible.

This example emphasizes:

1. Unsupervised dimensionality reduction.
2. Standardization before PCA.
3. Explained variance and explained variance ratio.
4. Two-dimensional visualization.
5. Reconstruction error.
6. Interpretation of PCA projections.
7. Limitations of unsupervised projections in biomedical-style data.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Problem Statement

Principal Component Analysis, or PCA, is an unsupervised dimensionality reduction method.

The Breast Cancer dataset contains 30 numeric features. PCA creates new variables called principal components, which are linear combinations of the original features.

The first principal component captures the largest possible variance in the data. The second captures the largest remaining variance subject to being orthogonal to the first, and so on.

The main questions in this notebook are:

> How much of the original feature variation can be summarized by a small number of principal components?

> Does a two-dimensional PCA projection reveal structure related to malignant and benign samples?

The diagnosis labels are used only for visualization and interpretation. They are not used when fitting PCA.
"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import sys

# Make the local package importable whether the notebook is run from the
# project root or from inside the examples directory.
current_path = Path.cwd().resolve()
for candidate in [current_path, *current_path.parents]:
    if (candidate / "src" / "jiayi_ml").exists():
        sys.path.insert(0, str(candidate / "src"))
        PROJECT_ROOT = candidate
        break

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from jiayi_ml.preprocessing import StandardScaler
from jiayi_ml.unsupervised import PCA
from jiayi_ml.metrics import mean_squared_error, root_mean_squared_error

np.random.seed(438)
pd.set_option("display.precision", 4)
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 2. Load the Dataset

The dataset is loaded from `sklearn.datasets.load_breast_cancer`, so this notebook is reproducible without external downloads.

The dataset contains numeric features computed from digitized images of breast mass cell nuclei.

The original sklearn target encoding is:

- `0` = malignant
- `1` = benign

For this notebook, we recode the labels only for easier interpretation:

- `1` = malignant
- `0` = benign

These labels are not used in PCA fitting.
"""
    ),
    nbf.v4.new_code_cell(
        """data = load_breast_cancer(as_frame=True)

X = data.data
y_original = data.target
feature_names = X.columns.tolist()
target_names = data.target_names

# Recode target for interpretation:
# original 0 = malignant, original 1 = benign
# new 1 = malignant, new 0 = benign
y = (y_original == 0).astype(int)

df = X.copy()
df["malignant"] = y

print("Feature matrix shape:", X.shape)
print("Number of features:", len(feature_names))
print("Original target names:", target_names)

df.head()
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 3. Exploratory Data Analysis

Before applying PCA, we inspect:

- Missing values
- Feature summary statistics
- Class distribution for later visualization
- Feature scale differences

PCA is sensitive to feature scale because it is based on variance. Features with larger numeric ranges can dominate the principal components if the data is not standardized.
"""
    ),
    nbf.v4.new_code_cell(
        """missing_values = df.isna().sum()

print("Total missing values:", int(missing_values.sum()))

summary = X.describe().T
summary.head(12)
"""
    ),
    nbf.v4.new_code_cell(
        """class_counts = pd.Series(y).value_counts().sort_index()
class_counts.index = ["benign (0)", "malignant (1)"]

class_counts_df = class_counts.to_frame(name="count")
class_counts_df["proportion"] = class_counts_df["count"] / class_counts_df["count"].sum()
class_counts_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(6, 4))
plt.bar(class_counts_df.index, class_counts_df["count"])
plt.ylabel("Number of samples")
plt.title("Class Distribution")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """feature_ranges = (X.max() - X.min()).sort_values(ascending=False)

plt.figure(figsize=(10, 4))
plt.bar(feature_ranges.index[:12], feature_ranges.values[:12])
plt.xticks(rotation=60, ha="right")
plt.ylabel("Feature range")
plt.title("Largest Feature Ranges Before Standardization")
plt.tight_layout()
plt.show()

feature_ranges.head(12).to_frame(name="range")
"""
    ),
    nbf.v4.new_markdown_cell(
        """The feature ranges differ substantially. This confirms that standardization is necessary before PCA. Without standardization, high-range variables could dominate the components even if they are not intrinsically more informative.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Standardization

PCA is fitted on standardized features.

Each feature is centered and scaled to have mean 0 and standard deviation 1. This makes the PCA directions depend on correlation structure rather than raw feature scale.
"""
    ),
    nbf.v4.new_code_cell(
        """scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

scaled_summary = pd.DataFrame(X_scaled, columns=feature_names).describe().T
scaled_summary[["mean", "std"]].head(10)
"""
    ),
    nbf.v4.new_markdown_cell(
        """The standardized features have approximately mean 0 and standard deviation 1. This makes the features comparable before dimensionality reduction.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. Fit PCA with All Components

We first fit PCA while keeping all possible components. This allows us to inspect the full explained variance pattern.

The labels are not used when fitting PCA.
"""
    ),
    nbf.v4.new_code_cell(
        """pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

print("PCA transformed shape:", X_pca_full.shape)
print("Components shape:", pca_full.components_.shape)
print("Explained variance ratio shape:", pca_full.explained_variance_ratio_.shape)
print("Sum of explained variance ratio:", np.sum(pca_full.explained_variance_ratio_))
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 6. Explained Variance

The explained variance ratio tells us what proportion of total variance is captured by each principal component.

The cumulative explained variance shows how many components are needed to capture a given amount of variation.
"""
    ),
    nbf.v4.new_code_cell(
        """explained_df = pd.DataFrame(
    {
        "component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
        "explained_variance": pca_full.explained_variance_,
        "explained_variance_ratio": pca_full.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca_full.explained_variance_ratio_),
    }
)

explained_df.head(12)
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.bar(
    explained_df["component"][:15],
    explained_df["explained_variance_ratio"][:15],
)
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio")
plt.title("Explained Variance Ratio by Principal Component")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.plot(
    explained_df["component"],
    explained_df["cumulative_explained_variance_ratio"],
    marker="o",
)
plt.axhline(0.80, linestyle="--", label="80% variance")
plt.axhline(0.90, linestyle="--", label="90% variance")
plt.axhline(0.95, linestyle="--", label="95% variance")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance ratio")
plt.title("Cumulative Explained Variance")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """for threshold in [0.80, 0.90, 0.95]:
    n_needed = int(np.argmax(explained_df["cumulative_explained_variance_ratio"].values >= threshold) + 1)
    print(f"Components needed for {int(threshold * 100)}% variance: {n_needed}")
"""
    ),
    nbf.v4.new_markdown_cell(
        """The explained variance analysis shows how much dimensionality reduction is possible. If a small number of components captures most of the variance, PCA can provide a compact summary of the dataset.

However, variance is not the same as predictive usefulness. A component can explain substantial variance without being the best direction for class separation.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 7. Two-Dimensional PCA Projection

Next, we fit PCA with two components and visualize the data in two dimensions.

The diagnosis label is used only for coloring the plot after PCA has already been fit. PCA itself does not use the label.
"""
    ),
    nbf.v4.new_code_cell(
        """pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

pca_2d_df = pd.DataFrame(
    X_pca_2d,
    columns=["PC1", "PC2"],
)
pca_2d_df["malignant"] = y
pca_2d_df["label"] = pca_2d_df["malignant"].map({0: "benign", 1: "malignant"})

print("2D PCA shape:", X_pca_2d.shape)
print("Explained variance ratio:", pca_2d.explained_variance_ratio_)
print("Total variance explained by first two PCs:", np.sum(pca_2d.explained_variance_ratio_))

pca_2d_df.head()
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(7, 6))

for label_value, label_name in [(0, "benign"), (1, "malignant")]:
    mask = pca_2d_df["malignant"] == label_value
    plt.scatter(
        pca_2d_df.loc[mask, "PC1"],
        pca_2d_df.loc[mask, "PC2"],
        alpha=0.75,
        edgecolors="k",
        label=label_name,
    )

plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.title("Two-Dimensional PCA Projection")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The two-dimensional PCA projection can reveal whether high-dimensional feature variation aligns with diagnosis-related structure.

If malignant and benign samples show visible separation, this suggests that the high-dimensional measurements contain structure related to diagnosis. However, PCA is unsupervised, so the components are not optimized for classification.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 8. Component Loadings

Each principal component is a linear combination of the original standardized features.

The loading values show how strongly each original feature contributes to a component. Large positive or negative loadings indicate features that strongly influence the component direction.
"""
    ),
    nbf.v4.new_code_cell(
        """loadings_df = pd.DataFrame(
    pca_2d.components_.T,
    index=feature_names,
    columns=["PC1_loading", "PC2_loading"],
)

loadings_df["abs_PC1_loading"] = loadings_df["PC1_loading"].abs()
loadings_df["abs_PC2_loading"] = loadings_df["PC2_loading"].abs()

loadings_df.sort_values("abs_PC1_loading", ascending=False).head(12)
"""
    ),
    nbf.v4.new_code_cell(
        """top_pc1 = loadings_df.sort_values("abs_PC1_loading", ascending=False).head(12)
top_pc1_plot = top_pc1.sort_values("PC1_loading")

plt.figure(figsize=(8, 5))
plt.barh(top_pc1_plot.index, top_pc1_plot["PC1_loading"])
plt.axvline(0, linewidth=1)
plt.xlabel("PC1 loading")
plt.title("Largest Feature Loadings on PC1")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """top_pc2 = loadings_df.sort_values("abs_PC2_loading", ascending=False).head(12)
top_pc2_plot = top_pc2.sort_values("PC2_loading")

plt.figure(figsize=(8, 5))
plt.barh(top_pc2_plot.index, top_pc2_plot["PC2_loading"])
plt.axvline(0, linewidth=1)
plt.xlabel("PC2 loading")
plt.title("Largest Feature Loadings on PC2")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The loading plots help interpret what each component represents. For example, if several size-related features have large loadings on PC1, then PC1 may summarize a general tumor-size pattern.

This interpretation is descriptive. PCA components are mathematical directions of variance, not causal biological factors.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 9. Reconstruction from Principal Components

PCA can also be viewed as compression. If we keep only a few components, we can reconstruct an approximation of the original standardized data.

The reconstruction error measures how much information is lost when reducing dimensionality.
"""
    ),
    nbf.v4.new_code_cell(
        """component_values = [2, 5, 10, 15, 20, 30]
reconstruction_results = []

for n_components in component_values:
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_reduced)
    
    reconstruction_results.append(
        {
            "n_components": n_components,
            "cumulative_explained_variance": np.sum(pca.explained_variance_ratio_),
            "MSE": mean_squared_error(X_scaled.ravel(), X_reconstructed.ravel()),
            "RMSE": root_mean_squared_error(X_scaled.ravel(), X_reconstructed.ravel()),
        }
    )

reconstruction_df = pd.DataFrame(reconstruction_results)
reconstruction_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.plot(
    reconstruction_df["n_components"],
    reconstruction_df["RMSE"],
    marker="o",
)
plt.xlabel("Number of components")
plt.ylabel("Reconstruction RMSE")
plt.title("Reconstruction Error vs. Number of Components")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.plot(
    reconstruction_df["n_components"],
    reconstruction_df["cumulative_explained_variance"],
    marker="o",
)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Variance Retained vs. Number of Components")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """As the number of components increases, reconstruction error decreases and explained variance increases.

This illustrates the trade-off in dimensionality reduction:

- fewer components improve compression and visualization,
- more components preserve more information.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 10. PCA Is Not Classification

Although the PCA projection is colored by diagnosis labels, PCA itself is not a classifier.

PCA does not use labels. It finds directions that explain variance in the feature matrix, not directions that maximize class separation.

A supervised method such as logistic regression, decision trees, or random forests should be used when the goal is prediction. PCA is more appropriate for exploration, visualization, compression, and preprocessing.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 11. Interpretation

PCA provides a lower-dimensional view of the Breast Cancer dataset. The first few components capture a substantial amount of feature variation, and the two-dimensional projection may show structure related to malignant and benign samples.

The component loadings suggest which original features contribute most strongly to the principal directions.

However, the components should be interpreted carefully. They are linear combinations chosen to explain variance, not necessarily clinical or biological mechanisms.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 12. Limitations

This analysis has several limitations:

1. PCA is linear and may miss nonlinear structure.
2. PCA is unsupervised and does not optimize class separation.
3. Components can be difficult to interpret because they combine many original features.
4. Results depend on feature scaling.
5. High explained variance does not necessarily imply high predictive performance.
6. The two-dimensional plot is only a projection of a 30-dimensional dataset.
7. This notebook is an educational dimensionality reduction example, not a clinical tool.

A stronger analysis could compare PCA with supervised classifiers, nonlinear dimensionality reduction methods, and cross-validated downstream performance.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 13. Conclusion

This notebook demonstrated PCA using the custom `PCA` implementation from `jiayi_ml`.

Key takeaways:

- PCA is an unsupervised dimensionality reduction method.
- Standardization is important before PCA.
- Explained variance measures how much variation each component captures.
- A two-dimensional PCA projection can help visualize high-dimensional data.
- Component loadings help interpret principal components.
- Reconstruction error quantifies information loss from dimensionality reduction.
- PCA is useful for exploration and compression, but it is not a classifier.
"""
    ),
]

nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written to: {NOTEBOOK_PATH}")