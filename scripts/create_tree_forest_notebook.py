from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "examples"
    / "supervised_learning"
    / "04_decision_tree_random_forest_wine.ipynb"
)

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Decision Tree and Random Forest on the Wine Dataset

This notebook demonstrates tree-based classification models implemented in the `jiayi_ml` package:

- `DecisionTreeClassifier`
- `RandomForestClassifier`

The goal is to classify wine samples into one of three cultivars based on chemical measurements.

This example emphasizes:

1. Multiclass classification.
2. Decision tree interpretability.
3. Overfitting in single trees.
4. Ensemble stability in random forests.
5. Feature importance analysis.
6. Confusion matrix interpretation.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Problem Statement

The Wine dataset is a supervised multiclass classification dataset.

Each observation is a wine sample described by chemical measurements. The target label represents the wine cultivar.

The prediction task is:

> Given chemical measurements of a wine sample, predict its cultivar class.

This notebook compares two tree-based approaches:

- A single decision tree, which is easy to interpret but can overfit.
- A random forest, which averages many trees to improve stability and generalization.

The main modeling question is:

> Does the ensemble model improve generalization compared with a single decision tree?
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
from sklearn.model_selection import train_test_split

from jiayi_ml.preprocessing import StandardScaler
from jiayi_ml.supervised import DecisionTreeClassifier, RandomForestClassifier
from jiayi_ml.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

np.random.seed(438)
pd.set_option("display.precision", 4)
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 2. Load the Dataset

The Wine dataset is loaded from `sklearn.datasets.load_wine`, so this notebook is reproducible without external downloads.

The dataset contains 13 chemical measurements, including alcohol, malic acid, ash, alkalinity of ash, magnesium, phenols, flavanoids, color intensity, hue, and proline.

The target has three classes, corresponding to three wine cultivars.
"""
    ),
    nbf.v4.new_code_cell(
        """wine = load_wine(as_frame=True)

X = wine.data
y = wine.target
feature_names = X.columns.tolist()
target_names = wine.target_names

df = X.copy()
df["class"] = y
df["class_name"] = df["class"].map(lambda idx: target_names[idx])

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("Number of features:", len(feature_names))
print("Target names:", target_names)

df.head()
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 3. Exploratory Data Analysis

Before modeling, we inspect missing values, feature summaries, and class balance.

Class balance matters because a model can achieve misleadingly high accuracy if one class dominates the dataset. Here, the three classes are not exactly equal but all are represented.
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
        """class_counts = df["class_name"].value_counts().reindex(target_names)

class_counts_df = class_counts.to_frame(name="count")
class_counts_df["proportion"] = class_counts_df["count"] / class_counts_df["count"].sum()
class_counts_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(6, 4))
plt.bar(class_counts_df.index, class_counts_df["count"])
plt.ylabel("Number of samples")
plt.title("Wine Class Distribution")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Feature Exploration

Tree-based models can handle features with different scales because splits are based on thresholds. However, exploratory plots are still useful for understanding which features may help separate classes.

The following plots show the distributions of several chemical measurements by class.
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
            X.loc[y == class_idx, feature],
            bins=15,
            alpha=0.6,
            label=class_name,
        )
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature} by Wine Class")
    plt.legend()
    plt.tight_layout()
    plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """Some features show clear separation between classes, while others overlap. This motivates tree-based models because trees can create nonlinear decision rules through feature thresholds.

Unlike logistic regression, a decision tree does not require a single global linear boundary.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. Train/Test Split

The data is split into training and test sets using stratification to preserve class proportions.

Although tree-based models do not require standardization, this notebook still creates a standardized version of the data for consistency with other examples. The tree models below use the original feature values so that thresholds and feature importances remain easier to interpret.
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

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Training class counts:", np.bincount(y_train))
print("Test class counts:", np.bincount(y_test))
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 6. Fit Decision Tree and Random Forest Models

We fit three models:

1. A shallow decision tree.
2. A deeper decision tree.
3. A random forest.

The shallow tree is more constrained and easier to interpret. The deeper tree is more flexible and may overfit. The random forest averages multiple trees trained on bootstrap samples, which often improves stability.
"""
    ),
    nbf.v4.new_code_cell(
        """models = {
    "Decision Tree, max_depth=3": DecisionTreeClassifier(
        max_depth=3,
        criterion="gini",
        random_state=438,
    ),
    "Decision Tree, unrestricted": DecisionTreeClassifier(
        max_depth=None,
        criterion="gini",
        random_state=438,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        criterion="gini",
        max_features="sqrt",
        bootstrap=True,
        random_state=438,
    ),
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{model_name} fitted.")
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 7. Test-Set Evaluation

For multiclass classification, we evaluate:

- Accuracy
- Macro precision
- Macro recall
- Macro F1 score

Macro averaging computes the metric separately for each class and then averages across classes. This treats all classes equally.
"""
    ),
    nbf.v4.new_code_cell(
        """def macro_metric(metric_function, y_true, y_pred, labels):
    values = []
    for label in labels:
        values.append(metric_function(y_true, y_pred, pos_label=label))
    return float(np.mean(values))


labels = np.array([0, 1, 2])
results = []

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    
    results.append(
        {
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_precision": macro_metric(precision_score, y_test, y_pred, labels),
            "macro_recall": macro_metric(recall_score, y_test, y_pred, labels),
            "macro_f1": macro_metric(f1_score, y_test, y_pred, labels),
        }
    )

results_df = pd.DataFrame(results).set_index("model")
results_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.bar(results_df.index, results_df["accuracy"])
plt.ylabel("Test accuracy")
plt.title("Decision Tree vs. Random Forest Test Accuracy")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """A single unrestricted tree can fit complex patterns but may be unstable because small changes in the training data can produce different splits. A random forest reduces this instability by averaging predictions across many trees.

The test-set metrics allow us to compare whether the ensemble improves generalization relative to a single tree.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 8. Confusion Matrices

Confusion matrices show which wine classes are confused by each model.

This is important because two models may have similar accuracy but make different types of mistakes.
"""
    ),
    nbf.v4.new_code_cell(
        """for model_name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    
    cm_df = pd.DataFrame(
        cm,
        index=[f"true {name}" for name in target_names],
        columns=[f"predicted {name}" for name in target_names],
    )
    
    print(model_name)
    display(cm_df)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.xticks([0, 1, 2], target_names, rotation=30)
    plt.yticks([0, 1, 2], target_names)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The confusion matrices identify class-specific errors. If most mistakes occur between two particular cultivars, that suggests those classes have overlapping chemical profiles in the selected feature space.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 9. Feature Importance

Both the custom decision tree and random forest implementations compute feature importances based on impurity reduction.

Feature importance answers:

> Which features were most useful for splitting the data?

For a single tree, importance can be unstable because it depends on one fitted tree structure. For a random forest, feature importance is averaged across many trees and is usually more stable.
"""
    ),
    nbf.v4.new_code_cell(
        """importance_frames = []

for model_name, model in models.items():
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
            "model": model_name,
        }
    ).sort_values("importance", ascending=False)
    
    importance_frames.append(importance_df)
    
    print(model_name)
    display(importance_df.head(8))
"""
    ),
    nbf.v4.new_code_cell(
        """forest_importance = importance_frames[-1].head(10).sort_values("importance")

plt.figure(figsize=(8, 5))
plt.barh(forest_importance["feature"], forest_importance["importance"])
plt.xlabel("Feature importance")
plt.title("Top Random Forest Feature Importances")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """Feature importance can help interpret which chemical measurements drive the model. However, it should be interpreted carefully:

- Correlated features can share or distort importance.
- Importance does not imply causality.
- A feature can be useful for prediction without being independently meaningful.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 10. Train vs. Test Performance

Overfitting can be assessed by comparing training and test performance.

A model with much higher training accuracy than test accuracy may be fitting noise or idiosyncrasies in the training set.
"""
    ),
    nbf.v4.new_code_cell(
        """train_test_comparison = []

for model_name, model in models.items():
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_test_comparison.append(
        {
            "model": model_name,
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "gap": accuracy_score(y_train, train_pred) - accuracy_score(y_test, test_pred),
        }
    )

train_test_df = pd.DataFrame(train_test_comparison).set_index("model")
train_test_df
"""
    ),
    nbf.v4.new_code_cell(
        """x = np.arange(len(train_test_df.index))
width = 0.35

plt.figure(figsize=(9, 4))
plt.bar(x - width / 2, train_test_df["train_accuracy"], width, label="train")
plt.bar(x + width / 2, train_test_df["test_accuracy"], width, label="test")
plt.xticks(x, train_test_df.index, rotation=20, ha="right")
plt.ylabel("Accuracy")
plt.title("Train vs. Test Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The train-test comparison illustrates the bias-variance trade-off.

A shallow tree may underfit but is more constrained. An unrestricted tree can fit training data very well but may generalize less reliably. A random forest often improves generalization by averaging multiple high-variance trees.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 11. Interpretation

The decision tree provides a transparent model structure based on threshold splits. This makes it useful for explanation, but it can be sensitive to small changes in the data.

The random forest sacrifices some interpretability at the individual-tree level but often improves predictive stability. Its feature importance values provide a summary of which variables were repeatedly useful across trees.

For the Wine dataset, tree-based models are appropriate because the classes can be separated using threshold-based rules on chemical measurements.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 12. Limitations

This analysis has several limitations:

1. The dataset is small.
2. The test set is also small, so performance estimates may vary with the split.
3. Hyperparameters were chosen for demonstration rather than tuned by cross-validation.
4. Feature importance is not causal importance.
5. A random forest is less directly interpretable than a single decision tree.
6. The custom implementation is educational and does not include all optimizations in production libraries.

A stronger analysis could include cross-validation, hyperparameter tuning, repeated train/test splits, and comparison with additional classifiers.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 13. Conclusion

This notebook demonstrated tree-based classification using custom implementations from `jiayi_ml`.

Key takeaways:

- Decision trees are interpretable but can overfit.
- Tree depth controls model complexity.
- Random forests improve stability by averaging many trees.
- Confusion matrices reveal class-specific mistakes.
- Feature importances summarize which variables were useful for splitting.
- Train/test comparison helps diagnose overfitting.
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