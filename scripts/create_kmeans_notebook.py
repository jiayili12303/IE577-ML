from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "examples"
    / "unsupervised_learning"
    / "07_kmeans_blobs.ipynb"
)

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# K-Means Clustering on Synthetic Blob Data

This notebook demonstrates the `KMeans` clustering algorithm implemented in the `jiayi_ml` package.

The goal is to identify groups in unlabeled data using distance-based clustering.

This example emphasizes:

1. Unsupervised learning.
2. Cluster visualization.
3. The role of the number of clusters.
4. K-Means inertia.
5. The elbow method.
6. Sensitivity to cluster shape and initialization.
7. Interpretation of clustering results without using labels for training.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Problem Statement

K-Means is an unsupervised clustering algorithm.

Unlike supervised learning, clustering does not use target labels during fitting. The goal is to discover structure in the feature space.

K-Means tries to partition data into `k` clusters by minimizing the total squared distance between each point and its assigned cluster center.

The main question in this notebook is:

> Can K-Means recover meaningful cluster structure from unlabeled data?

A synthetic blob dataset is used because it allows clear visualization of the algorithm's strengths and limitations.
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

from sklearn.datasets import make_blobs

from jiayi_ml.preprocessing import StandardScaler
from jiayi_ml.unsupervised import KMeans

np.random.seed(438)
pd.set_option("display.precision", 4)
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 2. Generate Synthetic Blob Data

The dataset is generated with `sklearn.datasets.make_blobs`.

Although the generator provides true labels, these labels are not used during K-Means fitting. They are used only after clustering to help interpret whether the discovered clusters align with the known synthetic groups.

The data is suitable for K-Means because blob-shaped clusters are roughly spherical and separated in Euclidean space.
"""
    ),
    nbf.v4.new_code_cell(
        """X, true_labels = make_blobs(
    n_samples=450,
    centers=4,
    cluster_std=[0.8, 1.1, 0.7, 1.0],
    random_state=438,
)

df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
df["true_group"] = true_labels

print("Feature matrix shape:", X.shape)
print("Number of synthetic groups:", len(np.unique(true_labels)))

df.head()
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 3. Exploratory Visualization

Because this dataset has two features, we can directly visualize the samples.

The first plot shows the unlabeled data, which is the view available to K-Means. The second plot uses the synthetic labels only for interpretation.
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], alpha=0.75, edgecolors="k")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title("Unlabeled Synthetic Blob Data")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=true_labels, alpha=0.75, edgecolors="k")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title("Synthetic Data Colored by True Group")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The clusters appear compact and separated, which matches the assumptions of K-Means. In real applications, true labels are usually unavailable, so the first plot is the more realistic unsupervised learning setting.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Standardization

K-Means relies on Euclidean distances. If features have very different scales, features with larger numeric ranges can dominate the clustering result.

Here, both features are generated on comparable scales, but we still standardize to demonstrate a consistent preprocessing workflow.
"""
    ),
    nbf.v4.new_code_cell(
        """scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(6, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.75, edgecolors="k")
plt.xlabel("standardized feature 1")
plt.ylabel("standardized feature 2")
plt.title("Standardized Blob Data")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. Fit K-Means with the Correct Number of Clusters

We first fit K-Means with `n_clusters = 4`, matching the known number of synthetic centers.

The algorithm does not know the true labels. It only sees the feature matrix.
"""
    ),
    nbf.v4.new_code_cell(
        """model = KMeans(
    n_clusters=4,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=438,
)

cluster_labels = model.fit_predict(X_scaled)

print("Cluster centers shape:", model.cluster_centers_.shape)
print("Inertia:", model.inertia_)
print("Iterations for best run:", model.n_iter_)
print("Cluster counts:", np.bincount(cluster_labels))
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(6, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, alpha=0.75, edgecolors="k")
plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=220,
    edgecolors="k",
    label="cluster centers",
)
plt.xlabel("standardized feature 1")
plt.ylabel("standardized feature 2")
plt.title("K-Means Clustering Result with k=4")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The cluster centers summarize the location of each group. Each sample is assigned to the nearest center.

For this blob-shaped dataset, K-Means is expected to perform well because the clusters are compact and approximately spherical.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 6. Compare Cluster Labels with True Synthetic Groups

In real unsupervised learning, true labels are often unavailable. In this synthetic dataset, true labels are available only for evaluation and interpretation.

The contingency table below compares discovered clusters with true synthetic groups.
"""
    ),
    nbf.v4.new_code_cell(
        """comparison_df = pd.DataFrame(
    {
        "true_group": true_labels,
        "kmeans_cluster": cluster_labels,
    }
)

contingency_table = pd.crosstab(
    comparison_df["true_group"],
    comparison_df["kmeans_cluster"],
    rownames=["true group"],
    colnames=["K-Means cluster"],
)

contingency_table
"""
    ),
    nbf.v4.new_markdown_cell(
        """A strong clustering result should show that each true group mostly maps to one discovered cluster. The numeric cluster labels themselves are arbitrary, so cluster 0 does not need to correspond to true group 0.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 7. Choosing k with the Elbow Method

In real clustering problems, the correct number of clusters is usually unknown.

The elbow method fits K-Means for several values of `k` and plots inertia. Inertia decreases as `k` increases, because more clusters can always reduce within-cluster distances.

The goal is to look for a point where the improvement begins to slow down.
"""
    ),
    nbf.v4.new_code_cell(
        """k_values = list(range(1, 11))
inertias = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=438,
    )
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

elbow_df = pd.DataFrame(
    {
        "k": k_values,
        "inertia": inertias,
    }
)

elbow_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(7, 4))
plt.plot(elbow_df["k"], elbow_df["inertia"], marker="o")
plt.xlabel("Number of clusters k")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means")
plt.xticks(k_values)
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The elbow plot should show a sharp decrease in inertia when moving from too few clusters toward the true cluster structure. After the main cluster structure is captured, additional clusters reduce inertia more slowly.

The elbow method is useful but subjective. It should be combined with domain knowledge and visualization when possible.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 8. Effect of Choosing the Wrong Number of Clusters

K-Means requires the user to specify `k`. If `k` is too small, distinct groups may be merged. If `k` is too large, natural groups may be split into artificial subclusters.
"""
    ),
    nbf.v4.new_code_cell(
        """for k in [2, 3, 4, 6]:
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=438,
    )
    labels = kmeans.fit_predict(X_scaled)
    
    plt.figure(figsize=(6, 5))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, alpha=0.75, edgecolors="k")
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker="X",
        s=220,
        edgecolors="k",
    )
    plt.xlabel("standardized feature 1")
    plt.ylabel("standardized feature 2")
    plt.title(f"K-Means Clustering with k={k}")
    plt.tight_layout()
    plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """This comparison illustrates that K-Means will always return the requested number of clusters, even if that number does not reflect the natural structure of the data. This is both a strength and a limitation.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 9. Sensitivity to Cluster Shape

K-Means works best when clusters are:

- roughly spherical
- similarly sized
- separated by Euclidean distance
- not strongly affected by outliers

When clusters are non-spherical, have different densities, or contain noise, methods such as DBSCAN may be more appropriate. This limitation is explored in the DBSCAN notebook.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 10. Interpretation

K-Means successfully identifies the main groups in this synthetic blob dataset because the data structure matches the algorithm's assumptions.

The cluster centers provide a compact summary of the data. However, the clusters should be interpreted as distance-based groups, not necessarily as meaningful real-world categories unless supported by domain knowledge.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 11. Limitations

This analysis has several limitations:

1. The dataset is synthetic and cleaner than many real datasets.
2. K-Means requires the number of clusters to be specified in advance.
3. The algorithm assumes Euclidean distance is meaningful.
4. It works best for compact, spherical clusters.
5. It can be sensitive to initialization, although k-means++ and multiple initializations reduce this issue.
6. It does not naturally identify noise points.
7. Cluster labels are arbitrary and require interpretation.

A stronger real-world analysis would combine clustering with domain knowledge, stability checks, and external validation if labels are available.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 12. Conclusion

This notebook demonstrated K-Means clustering using the custom `KMeans` implementation from `jiayi_ml`.

Key takeaways:

- K-Means is an unsupervised clustering algorithm.
- It assigns points to the nearest cluster center.
- Inertia measures within-cluster compactness.
- The elbow method can help choose the number of clusters.
- K-Means performs well on compact blob-shaped clusters.
- K-Means can perform poorly when cluster shapes are non-spherical or when noise is present.
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