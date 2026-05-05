from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "examples"
    / "supervised_learning"
    / "03_knn_iris.ipynb"
)

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# K-Nearest Neighbors on the Iris Dataset

This notebook demonstrates two K-nearest neighbors models implemented in the `jiayi_ml` package:

- `KNNClassifier`
- `KNNRegressor`

The main classification example uses the Iris dataset. The notebook also includes a simple one-dimensional regression example to show how KNN regression behaves for different values of `k`.

This example emphasizes:

1. Distance-based learning.
2. The effect of the number of neighbors.
3. Feature scaling.
4. Multiclass classification evaluation.
5. Decision boundary visualization.
6. KNN regression smoothing behavior.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Problem Statement

K-nearest neighbors is a non-parametric, instance-based learning algorithm.

Unlike linear regression or logistic regression, KNN does not learn a global equation during training. Instead, it stores the training data and makes predictions based on the closest training samples.

For classification:

> A test point is assigned the most common class among its nearest neighbors.

For regression:

> A test point is assigned the average target value among its nearest neighbors.

This notebook uses KNN for:

1. Multiclass classification on the Iris dataset.
2. One-dimensional regression on a synthetic nonlinear dataset.
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

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from jiayi_ml.preprocessing import StandardScaler
from jiayi_ml.supervised import KNNClassifier, KNNRegressor
from jiayi_ml.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

np.random.seed(438)
pd.set_option("display.precision", 4)
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 2. Load the Iris Dataset

The Iris dataset contains measurements from three iris flower species:

- setosa
- versicolor
- virginica

Each sample has four numeric features:

- sepal length
- sepal width
- petal length
- petal width

The task is a **supervised multiclass classification** problem because the target has three known classes.
"""
    ),
    nbf.v4.new_code_cell(
        """iris = load_iris(as_frame=True)

X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = X.columns.tolist()

df = X.copy()
df["species"] = y
df["species_name"] = df["species"].map(lambda idx: target_names[idx])

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("Feature names:", feature_names)
print("Target names:", target_names)

df.head()
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 3. Exploratory Data Analysis

KNN depends directly on distances between samples. Therefore, feature distributions and feature scales matter.

The Iris features are all measured in centimeters, but their ranges still differ. Scaling is useful because KNN can otherwise be dominated by features with larger numeric ranges.
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
        """class_counts = df["species_name"].value_counts().reindex(target_names)

class_counts_df = class_counts.to_frame(name="count")
class_counts_df["proportion"] = class_counts_df["count"] / class_counts_df["count"].sum()
class_counts_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(6, 4))
plt.bar(class_counts_df.index, class_counts_df["count"])
plt.ylabel("Number of samples")
plt.title("Iris Class Distribution")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """for feature in feature_names:
    plt.figure(figsize=(6, 4))
    for class_idx, class_name in enumerate(target_names):
        plt.hist(
            X.loc[y == class_idx, feature],
            bins=15,
            alpha=0.6,
            label=class_name,
        )
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature} by Species")
    plt.legend()
    plt.tight_layout()
    plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The class distributions are balanced, so accuracy is a reasonable first metric. However, because this is a multiclass classification problem, the confusion matrix is still important for identifying which species are confused with each other.

The feature histograms suggest that petal measurements are especially useful for separating species, while sepal measurements show more overlap.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Train/Test Split and Standardization

The data is split into training and test sets using stratification to preserve class proportions.

The scaler is fit only on the training set and then applied to the test set. This avoids data leakage.
"""
    ),
    nbf.v4.new_code_cell(
        """X_train, X_test, y_train, y_test = train_test_split(
    X.values,
    y.values,
    test_size=0.25,
    random_state=438,
    stratify=y.values,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training set shape:", X_train_scaled.shape)
print("Test set shape:", X_test_scaled.shape)
print("Training class counts:", np.bincount(y_train))
print("Test class counts:", np.bincount(y_test))
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. KNN Classification with Different Values of k

The number of neighbors controls the flexibility of the KNN classifier.

- Small `k`: more flexible, but more sensitive to noise.
- Large `k`: smoother decision boundary, but may underfit local structure.

The following experiment compares several values of `k` on the held-out test set.
"""
    ),
    nbf.v4.new_code_cell(
        """k_values = [1, 3, 5, 11, 21]
classification_results = []

trained_classifiers = {}

for k in k_values:
    model = KNNClassifier(n_neighbors=k, metric="euclidean", weights="uniform")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    trained_classifiers[k] = model
    
    classification_results.append(
        {
            "k": k,
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_precision": np.mean(
                [precision_score(y_test, y_pred, pos_label=label) for label in np.unique(y_test)]
            ),
            "macro_recall": np.mean(
                [recall_score(y_test, y_pred, pos_label=label) for label in np.unique(y_test)]
            ),
            "macro_f1": np.mean(
                [f1_score(y_test, y_pred, pos_label=label) for label in np.unique(y_test)]
            ),
        }
    )

classification_results_df = pd.DataFrame(classification_results)
classification_results_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(7, 4))
plt.plot(classification_results_df["k"], classification_results_df["accuracy"], marker="o", label="accuracy")
plt.plot(classification_results_df["k"], classification_results_df["macro_f1"], marker="o", label="macro F1")
plt.xlabel("Number of neighbors")
plt.ylabel("Metric value")
plt.title("KNN Classification Performance Across k")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The performance table shows how the model changes as `k` increases. If `k` is too small, predictions can be sensitive to individual training points. If `k` is too large, the model averages over too much of the dataset and can lose local class structure.

For this dataset, several values of `k` may perform well because Iris has strong feature separation, especially in petal measurements.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 6. Confusion Matrix for the Selected Model

We select `k = 5` as a reasonable middle value. It is flexible enough to capture local structure while being less sensitive to noise than `k = 1`.
"""
    ),
    nbf.v4.new_code_cell(
        """selected_k = 5
selected_model = trained_classifiers[selected_k]
y_pred_selected = selected_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_selected, labels=[0, 1, 2])
cm_df = pd.DataFrame(
    cm,
    index=[f"true {name}" for name in target_names],
    columns=[f"predicted {name}" for name in target_names],
)

cm_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.xticks([0, 1, 2], target_names, rotation=30)
plt.yticks([0, 1, 2], target_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title(f"Confusion Matrix for KNN Classifier, k={selected_k}")
plt.colorbar()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The confusion matrix identifies which species are most often confused. In many Iris analyses, setosa is easiest to separate, while versicolor and virginica can overlap more because their measurements are more similar.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 7. Decision Boundary Visualization Using Two Features

KNN decision boundaries are easiest to visualize in two dimensions.

The full model above used all four Iris features. For visualization only, this section uses:

- petal length
- petal width

These two features are chosen because the exploratory analysis suggests that petal measurements separate species well.
"""
    ),
    nbf.v4.new_code_cell(
        """petal_features = ["petal length (cm)", "petal width (cm)"]
X_petal = X[petal_features].values

X_petal_train, X_petal_test, y_petal_train, y_petal_test = train_test_split(
    X_petal,
    y.values,
    test_size=0.25,
    random_state=438,
    stratify=y.values,
)

petal_scaler = StandardScaler()
X_petal_train_scaled = petal_scaler.fit_transform(X_petal_train)
X_petal_test_scaled = petal_scaler.transform(X_petal_test)

x_min, x_max = X_petal_train_scaled[:, 0].min() - 0.5, X_petal_train_scaled[:, 0].max() + 0.5
y_min, y_max = X_petal_train_scaled[:, 1].min() - 0.5, X_petal_train_scaled[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 250),
    np.linspace(y_min, y_max, 250),
)

grid = np.c_[xx.ravel(), yy.ravel()]
"""
    ),
    nbf.v4.new_code_cell(
        """for k in [1, 5, 25]:
    boundary_model = KNNClassifier(n_neighbors=k)
    boundary_model.fit(X_petal_train_scaled, y_petal_train)
    grid_predictions = boundary_model.predict(grid).reshape(xx.shape)
    
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, grid_predictions, alpha=0.25)
    plt.scatter(
        X_petal_train_scaled[:, 0],
        X_petal_train_scaled[:, 1],
        c=y_petal_train,
        edgecolors="k",
        alpha=0.85,
    )
    plt.xlabel("standardized petal length")
    plt.ylabel("standardized petal width")
    plt.title(f"KNN Decision Boundary with k={k}")
    plt.tight_layout()
    plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The decision boundary visualization shows how `k` changes model flexibility.

- With `k = 1`, the boundary can be very jagged because each point has strong influence.
- With moderate `k`, the boundary becomes smoother.
- With very large `k`, the model may become too smooth and can ignore smaller local patterns.

This illustrates the bias-variance trade-off in KNN.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 8. KNN Regression on a Synthetic Nonlinear Dataset

KNN can also be used for regression.

To demonstrate this, we generate a one-dimensional nonlinear dataset. The true pattern is smooth, but the observations include noise.

This example is useful because it shows how the number of neighbors controls smoothing.
"""
    ),
    nbf.v4.new_code_cell(
        """rng = np.random.default_rng(438)

X_reg = np.linspace(0, 10, 120).reshape(-1, 1)
y_reg = np.sin(X_reg).ravel() + 0.25 * rng.normal(size=X_reg.shape[0])

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.30,
    random_state=438,
)

reg_scaler = StandardScaler()
X_reg_train_scaled = reg_scaler.fit_transform(X_reg_train)
X_reg_test_scaled = reg_scaler.transform(X_reg_test)

plt.figure(figsize=(7, 4))
plt.scatter(X_reg_train, y_reg_train, alpha=0.7, label="training data")
plt.scatter(X_reg_test, y_reg_test, alpha=0.7, label="test data")
plt.xlabel("x")
plt.ylabel("target")
plt.title("Synthetic Nonlinear Regression Dataset")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """regression_k_values = [1, 5, 15]
regression_results = []

x_grid_original = np.linspace(0, 10, 300).reshape(-1, 1)
x_grid_scaled = reg_scaler.transform(x_grid_original)

for k in regression_k_values:
    reg_model = KNNRegressor(n_neighbors=k)
    reg_model.fit(X_reg_train_scaled, y_reg_train)
    
    y_reg_pred = reg_model.predict(X_reg_test_scaled)
    y_grid_pred = reg_model.predict(x_grid_scaled)
    
    regression_results.append(
        {
            "k": k,
            "MSE": mean_squared_error(y_reg_test, y_reg_pred),
            "RMSE": root_mean_squared_error(y_reg_test, y_reg_pred),
            "MAE": mean_absolute_error(y_reg_test, y_reg_pred),
            "R2": r2_score(y_reg_test, y_reg_pred),
        }
    )
    
    plt.figure(figsize=(7, 4))
    plt.scatter(X_reg_train, y_reg_train, alpha=0.5, label="training data")
    plt.scatter(X_reg_test, y_reg_test, alpha=0.7, label="test data")
    plt.plot(x_grid_original, y_grid_pred, linewidth=2, label=f"KNN prediction, k={k}")
    plt.xlabel("x")
    plt.ylabel("target")
    plt.title(f"KNN Regression with k={k}")
    plt.legend()
    plt.tight_layout()
    plt.show()

regression_results_df = pd.DataFrame(regression_results)
regression_results_df
"""
    ),
    nbf.v4.new_markdown_cell(
        """The regression plots show the smoothing behavior of KNN.

- Small `k` follows local variation closely and may overfit noise.
- Larger `k` produces smoother predictions but may underfit sharp local structure.
- The best value of `k` depends on the noise level and the underlying function complexity.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 9. Interpretation

KNN is simple and intuitive, but it depends heavily on the distance metric, feature scaling, and the choice of `k`.

For the Iris classification task, KNN performs well because the classes are relatively well separated in the feature space.

For the synthetic regression task, KNN demonstrates local averaging. This makes it flexible, but also sensitive to the number of neighbors.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 10. Limitations

KNN has several important limitations:

1. It can be computationally expensive at prediction time because it compares each test point to training samples.
2. It is sensitive to feature scaling.
3. It can perform poorly in high-dimensional spaces because distances become less informative.
4. The choice of `k` strongly affects model behavior.
5. It does not provide coefficients or a compact parametric model.
6. It can be affected by irrelevant or noisy features.

Despite these limitations, KNN is a useful baseline and an intuitive example of distance-based learning.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 11. Conclusion

This notebook demonstrated KNN classification and regression using custom implementations from `jiayi_ml`.

Key takeaways:

- KNN is an instance-based learning algorithm.
- Feature scaling is important because KNN relies on distances.
- Smaller `k` values produce more flexible models.
- Larger `k` values produce smoother models.
- Decision boundaries help visualize how KNN behaves.
- KNN can be used for both classification and regression.
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