from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "examples"
    / "unsupervised_learning"
    / "09_hierarchical_clustering_wine.ipynb"
)

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Hierarchical Clustering on the Wine Dataset

This notebook demonstrates the `AgglomerativeClustering` algorithm implemented in the `jiayi_ml` package.

The goal is to discover groups of wine samples using chemical measurements without using the class labels during model fitting.

This example emphasizes:

1. Unsupervised learning on real tabular data.
2. Agglomerative bottom-up hierarchical clustering.
3. Linkage methods: single, complete, and average.
4. Feature scaling for distance-based clustering.
5. Cluster interpretation.
6. Comparison with known labels only after clustering.
7. Limitations of clustering evaluation.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Problem Statement

Hierarchical clustering is an unsupervised learning method.

Unlike classification, the algorithm does not use target labels during fitting. Instead, it attempts to discover structure based on distances between samples.

Agglomerative clustering follows a bottom-up strategy:

1. Start with each observation as its own cluster.
2. Repeatedly merge the two closest clusters.
3. Stop when the desired number of clusters remains.

The Wine dataset has known cultivar labels, but this notebook uses those labels only after clustering to interpret whether the discovered groups align with the known classes.

The main question is:

> Can hierarchical clustering recover meaningful group structure from wine chemical measurements?
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

from sklearn.datasets import load_wine

from jiayi_ml.preprocessing import StandardScaler
from jiayi_ml.unsupervised import AgglomerativeClustering

np.random.seed(438)
pd.set_option("display.precision", 4)
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 2. Load the Wine Dataset

The Wine dataset is loaded from `sklearn.datasets.load_wine`.

It contains chemical measurements for wine samples from three cultivars. The true class labels are available, but they are not used during clustering.

The features include measurements such as alcohol, malic acid, ash, magnesium, phenols, flavanoids, color intensity, hue, and proline.
"""
    ),
    nbf.v4.new_code_cell(
        """wine = load_wine(as_frame=True)

X = wine.data
true_labels = wine.target
feature_names = X.columns.tolist()
target_names = wine.target_names

df = X.copy()
df["true_class"] = true_labels
df["true_class_name"] = df["true_class"].map(lambda idx: target_names[idx])

print("Feature matrix shape:", X.shape)
print("Number of features:", len(feature_names))
print("Target names:", target_names)

df.head()
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 3. Exploratory Data Analysis

Although clustering is unsupervised, exploratory data analysis is still important.

We inspect:

- Missing values
- Feature ranges
- True class distribution for later interpretation
- Selected feature distributions

Feature scaling is especially important because hierarchical clustering relies on distances between samples.
"""
    ),
    nbf.v4.new_code_cell(
        """missing_values = df.isna().sum()

print("Total missing values:", int(missing_values.sum()))

summary = X.describe().T
summary
"""
    ),
    nbf.v4.new_code_cell(
        """class_counts = df["true_class_name"].value_counts().reindex(target_names)

class_counts_df = class_counts.to_frame(name="count")
class_counts_df["proportion"] = class_counts_df["count"] / class_counts_df["count"].sum()
class_counts_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(6, 4))
plt.bar(class_counts_df.index, class_counts_df["count"])
plt.ylabel("Number of samples")
plt.title("Known Wine Class Distribution")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The true class labels are shown only to understand the dataset. They will not be used when fitting the clustering model.
"""
    ),
    nbf.v4.new_code_cell(
        """selected_features = [
    "alcohol",
    "flavanoids",
    "color_intensity",
    "proline",
]

for feature in selected_features:
    plt.figure(figsize=(6, 4))
    for class_idx, class_name in enumerate(target_names):
        plt.hist(
            X.loc[true_labels == class_idx, feature],
            bins=15,
            alpha=0.6,
            label=class_name,
        )
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature} by Known Class")
    plt.legend()
    plt.tight_layout()
    plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """Several features show differences across known wine classes. This suggests that unsupervised clustering may be able to recover some class-related structure, even without using the labels during fitting.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Standardization

Hierarchical clustering depends on distances. If one feature has a much larger numeric range than another, it can dominate the distance calculation.

The Wine features are measured on different scales, so standardization is necessary before clustering.
"""
    ),
    nbf.v4.new_code_cell(
        """scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

scaled_df.describe().T.head()
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. Fit Agglomerative Clustering

We first fit agglomerative clustering with:

- `n_clusters = 3`, matching the known number of wine cultivars for interpretation.
- `linkage = "average"`, which uses the average distance between points in two clusters.
- `metric = "euclidean"`.

The true labels are not used in fitting.
"""
    ),
    nbf.v4.new_code_cell(
        """model = AgglomerativeClustering(
    n_clusters=3,
    linkage="average",
    metric="euclidean",
)

cluster_labels = model.fit_predict(X_scaled)

print("Cluster labels shape:", cluster_labels.shape)
print("Unique cluster labels:", np.unique(cluster_labels))
print("Cluster counts:", np.bincount(cluster_labels))
print("Children matrix shape:", model.children_.shape)
print("Distances shape:", model.distances_.shape)
"""
    ),
    nbf.v4.new_markdown_cell(
        """The `children_` attribute records which clusters were merged at each step. The `distances_` attribute records the distance between the merged clusters.

These values describe the hierarchy constructed by the algorithm.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 6. Cluster Sizes and Label Comparison

The numeric cluster IDs are arbitrary. Cluster 0 is not inherently better or earlier than cluster 1.

To interpret the result, we compare discovered clusters with the known wine classes after fitting.
"""
    ),
    nbf.v4.new_code_cell(
        """cluster_summary = pd.Series(cluster_labels).value_counts().sort_index()
cluster_summary_df = cluster_summary.to_frame(name="count")
cluster_summary_df["proportion"] = cluster_summary_df["count"] / cluster_summary_df["count"].sum()
cluster_summary_df
"""
    ),
    nbf.v4.new_code_cell(
        """comparison_df = pd.DataFrame(
    {
        "true_class": true_labels,
        "true_class_name": [target_names[idx] for idx in true_labels],
        "cluster": cluster_labels,
    }
)

contingency_table = pd.crosstab(
    comparison_df["true_class_name"],
    comparison_df["cluster"],
    rownames=["Known class"],
    colnames=["Hierarchical cluster"],
)

contingency_table
"""
    ),
    nbf.v4.new_markdown_cell(
        """A good clustering result would show that each known class mostly maps to one discovered cluster.

However, clustering is not classification. The algorithm did not know the true labels, and exact label recovery is not guaranteed.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 7. Visualize Clusters Using Two Features

The full clustering model uses all 13 standardized features. For visualization, we plot two informative features.

This two-dimensional plot is only a projection and does not show the full feature space used by the model.
"""
    ),
    nbf.v4.new_code_cell(
        """plot_features = ["flavanoids", "color_intensity"]
plot_indices = [feature_names.index(feature) for feature in plot_features]

plt.figure(figsize=(6, 5))
plt.scatter(
    X_scaled[:, plot_indices[0]],
    X_scaled[:, plot_indices[1]],
    c=cluster_labels,
    alpha=0.8,
    edgecolors="k",
)
plt.xlabel(f"standardized {plot_features[0]}")
plt.ylabel(f"standardized {plot_features[1]}")
plt.title("Hierarchical Clustering Result")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(6, 5))
plt.scatter(
    X_scaled[:, plot_indices[0]],
    X_scaled[:, plot_indices[1]],
    c=true_labels,
    alpha=0.8,
    edgecolors="k",
)
plt.xlabel(f"standardized {plot_features[0]}")
plt.ylabel(f"standardized {plot_features[1]}")
plt.title("Known Wine Classes")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """Comparing the cluster plot with the known-class plot can help interpret the discovered groups. However, because the clustering used all features, a two-feature visualization may not fully reflect the clustering decision.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 8. Compare Linkage Methods

The linkage method defines how distance between clusters is computed.

This notebook compares:

- `single`: minimum distance between points in two clusters.
- `complete`: maximum distance between points in two clusters.
- `average`: average distance between points in two clusters.

Different linkage methods can produce different cluster structures.
"""
    ),
    nbf.v4.new_code_cell(
        """linkage_methods = ["single", "complete", "average"]
linkage_results = []

for linkage in linkage_methods:
    linkage_model = AgglomerativeClustering(
        n_clusters=3,
        linkage=linkage,
        metric="euclidean",
    )
    labels = linkage_model.fit_predict(X_scaled)
    
    counts = np.bincount(labels)
    linkage_results.append(
        {
            "linkage": linkage,
            "cluster_counts": counts.tolist(),
            "last_merge_distance": linkage_model.distances_[-1] if len(linkage_model.distances_) > 0 else np.nan,
        }
    )
    
    comparison = pd.crosstab(
        true_labels,
        labels,
        rownames=["known class"],
        colnames=[f"{linkage} cluster"],
    )
    
    print(f"Linkage method: {linkage}")
    display(comparison)
    
    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_scaled[:, plot_indices[0]],
        X_scaled[:, plot_indices[1]],
        c=labels,
        alpha=0.8,
        edgecolors="k",
    )
    plt.xlabel(f"standardized {plot_features[0]}")
    plt.ylabel(f"standardized {plot_features[1]}")
    plt.title(f"Hierarchical Clustering with {linkage} Linkage")
    plt.tight_layout()
    plt.show()

linkage_results_df = pd.DataFrame(linkage_results)
linkage_results_df
"""
    ),
    nbf.v4.new_markdown_cell(
        """Single linkage can be sensitive to chaining, where clusters are merged through narrow bridges of points. Complete linkage tends to create more compact clusters. Average linkage often provides a compromise.

The best linkage method depends on the data geometry and the analysis goal.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 9. Merge Distance Pattern

A full dendrogram is a common visualization for hierarchical clustering. Our custom implementation stores merge distances, which can be plotted to show how cluster merge distances change over time.

Large jumps in merge distance can suggest meaningful separation between clusters.
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(model.distances_) + 1), model.distances_, marker="o", markersize=3)
plt.xlabel("Merge step")
plt.ylabel("Merge distance")
plt.title("Agglomerative Clustering Merge Distances")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
last_n = min(25, len(model.distances_))
plt.plot(
    np.arange(len(model.distances_) - last_n + 1, len(model.distances_) + 1),
    model.distances_[-last_n:],
    marker="o",
)
plt.xlabel("Merge step")
plt.ylabel("Merge distance")
plt.title("Last Merge Distances")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The final merge distances are especially useful because they show when larger clusters are being combined. A sharp increase may indicate that the algorithm is merging groups that were previously well separated.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 10. Effect of Number of Clusters

The number of clusters is another modeling choice.

Although the Wine dataset has three known classes, real clustering problems often do not have a known number of groups. Here we compare several values of `n_clusters`.
"""
    ),
    nbf.v4.new_code_cell(
        """n_cluster_values = [2, 3, 4, 5]

for n_clusters in n_cluster_values:
    cluster_model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="average",
        metric="euclidean",
    )
    labels = cluster_model.fit_predict(X_scaled)
    
    print(f"n_clusters = {n_clusters}")
    print("cluster counts:", np.bincount(labels))
    
    table = pd.crosstab(
        true_labels,
        labels,
        rownames=["known class"],
        colnames=[f"cluster"],
    )
    display(table)
    
    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_scaled[:, plot_indices[0]],
        X_scaled[:, plot_indices[1]],
        c=labels,
        alpha=0.8,
        edgecolors="k",
    )
    plt.xlabel(f"standardized {plot_features[0]}")
    plt.ylabel(f"standardized {plot_features[1]}")
    plt.title(f"Hierarchical Clustering with n_clusters={n_clusters}")
    plt.tight_layout()
    plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The number of clusters affects the granularity of the discovered structure. Too few clusters may merge distinct groups, while too many clusters may split a natural group into smaller subgroups.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 11. Interpretation

Hierarchical clustering provides a structured way to explore nested group relationships.

For the Wine dataset, the chemical measurements contain enough structure that discovered clusters can partially align with known cultivars. However, cluster labels should be interpreted as data-driven groupings, not automatically as true biological or chemical categories.

The comparison with known labels is useful here because the dataset includes ground truth, but in real unsupervised learning, such labels may not exist.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 12. Limitations

This analysis has several limitations:

1. The true labels are not used in fitting, so exact class recovery is not guaranteed.
2. Results depend on the distance metric.
3. Results depend on the linkage method.
4. Results depend on the number of clusters selected.
5. Hierarchical clustering can be computationally expensive for large datasets.
6. Two-dimensional visualizations do not fully represent the full 13-dimensional clustering.
7. Cluster interpretation requires domain knowledge.

A stronger analysis could include additional cluster validation metrics, stability analysis, and domain-specific interpretation of cluster profiles.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 13. Conclusion

This notebook demonstrated agglomerative hierarchical clustering using the custom `AgglomerativeClustering` implementation from `jiayi_ml`.

Key takeaways:

- Hierarchical clustering is an unsupervised method.
- Agglomerative clustering starts with individual samples and merges clusters.
- Linkage choice affects cluster structure.
- Standardization is important for distance-based clustering.
- Known labels can be used after clustering for interpretation, but not during fitting.
- Hierarchical clustering is useful for exploring nested group structure.
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