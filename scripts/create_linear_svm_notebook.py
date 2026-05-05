from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "examples"
    / "supervised_learning"
    / "11_linear_svm_breast_cancer.ipynb"
)

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Linear Support Vector Machine on the Breast Cancer Dataset

This notebook demonstrates the `LinearSVM` classifier implemented in the `jiayi_ml` package.

The goal is to classify breast tumor samples as malignant or benign using numeric features computed from digitized images of breast mass cell nuclei.

This example emphasizes:

1. Binary classification.
2. Margin-based learning.
3. Hinge loss.
4. Linear decision boundaries.
5. L2 regularization.
6. Confusion matrix interpretation.
7. Decision score analysis.
8. Limitations of linear SVMs compared with kernel SVMs.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Problem Statement

The Breast Cancer Wisconsin dataset is a binary classification dataset.

Each sample describes a breast tumor using numeric measurements derived from cell nuclei. The original target labels are:

- `malignant`
- `benign`

For this notebook, the target is recoded so that:

- `1` = malignant
- `0` = benign

This makes malignant tumors the positive class.

The prediction task is:

> Given numeric tumor measurements, predict whether a tumor sample is malignant or benign.

This is a supervised binary classification problem.

This notebook is an educational machine learning example. It should not be interpreted as a clinical diagnostic tool.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 2. Modeling Hypothesis

A linear support vector machine should perform well on this dataset because many breast cancer features show strong separation between malignant and benign samples.

The main hypothesis is:

> If the standardized features provide a roughly linearly separable representation of the two classes, then a linear SVM should achieve strong test-set classification performance by learning a maximum-margin separating hyperplane.

However, the model may still make errors because:

- some benign and malignant samples overlap in feature space,
- the decision boundary is linear,
- the dataset is a classic benchmark and may not represent external clinical data,
- the selected regularization strength affects the margin and error trade-off.
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
from sklearn.model_selection import train_test_split

from jiayi_ml.preprocessing import StandardScaler
from jiayi_ml.supervised import LinearSVM
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
        """## 3. Load the Dataset

The dataset is loaded from `sklearn.datasets.load_breast_cancer`, so the notebook is reproducible without external files.

The original sklearn target encoding is:

- `0` = malignant
- `1` = benign

This notebook recodes the target so that malignant is the positive class.
"""
    ),
    nbf.v4.new_code_cell(
        """data = load_breast_cancer(as_frame=True)

X = data.data
y_original = data.target
feature_names = X.columns.tolist()
target_names = data.target_names

# Recode target:
# original 0 = malignant, original 1 = benign
# new 1 = malignant, new 0 = benign
y = (y_original == 0).astype(int)

df = X.copy()
df["malignant"] = y

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("Original target names:", target_names)
print("Number of features:", len(feature_names))

df.head()
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Exploratory Data Analysis

Before fitting the SVM, we inspect:

- missing values,
- class balance,
- feature ranges,
- correlations with the malignant indicator.

Class balance matters because accuracy alone can be misleading when one class is much more common than the other.

Feature scale matters because a linear SVM uses dot products and margin distances. Standardization is therefore important.
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
        """The feature ranges differ substantially, which supports the decision to standardize the data before fitting the SVM.
"""
    ),
    nbf.v4.new_code_cell(
        """correlations = df.corr(numeric_only=True)["malignant"].drop("malignant")
top_correlations = correlations.reindex(
    correlations.abs().sort_values(ascending=False).index
).head(12)

top_correlations_df = top_correlations.to_frame(name="correlation_with_malignant")
top_correlations_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(9, 4))
plt.bar(top_correlations.index, top_correlations.values)
plt.axhline(0, linewidth=1)
plt.xticks(rotation=60, ha="right")
plt.ylabel("Correlation with malignant class")
plt.title("Top Feature Correlations with Malignancy")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The correlation analysis suggests that several features are strongly associated with the malignant class. This supports the modeling hypothesis that a linear classifier may be effective.

This is not a causal interpretation. Correlation only describes association in this dataset.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. Train/Test Split and Standardization

The data is split into training and test sets using stratification so that class proportions are preserved.

The scaler is fit only on the training data and then applied to the test data. This avoids data leakage.
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
        """## 6. Fit the Custom Linear SVM

The model below uses the custom `LinearSVM` implementation from the local `jiayi_ml` package.

This implementation optimizes a soft-margin linear SVM objective using hinge loss and L2 regularization.

The regularization strength `alpha` controls the trade-off between margin size and weight shrinkage.
"""
    ),
    nbf.v4.new_code_cell(
        """model = LinearSVM(
    learning_rate=0.01,
    max_iter=10000,
    alpha=0.01,
    fit_intercept=True,
    tol=1e-8,
)

model.fit(X_train_scaled, y_train)

print("Number of iterations:", model.n_iter_)
print("Final objective value:", model.losses_[-1])
print("Coefficient shape:", model.coef_.shape)
print("Intercept:", model.intercept_)
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(7, 4))
plt.plot(model.losses_)
plt.xlabel("Iteration")
plt.ylabel("SVM objective")
plt.title("Linear SVM Training Objective")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The training objective decreases as the model updates its coefficients. This provides a basic check that gradient descent is optimizing the hinge-loss objective.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 7. Test-Set Evaluation

The linear SVM predicts the class based on the sign of its decision score.

A positive score corresponds to the malignant class, and a negative score corresponds to the benign class.

We evaluate the model using:

- Accuracy
- Precision for malignant class
- Recall for malignant class
- F1 score for malignant class
- Confusion matrix
"""
    ),
    nbf.v4.new_code_cell(
        """y_pred = model.predict(X_test_scaled)
decision_scores = model.decision_function(X_test_scaled)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision_malignant": precision_score(y_test, y_pred, pos_label=1),
    "recall_malignant": recall_score(y_test, y_pred, pos_label=1),
    "f1_malignant": f1_score(y_test, y_pred, pos_label=1),
}

metrics_df = pd.DataFrame(metrics, index=["Linear SVM"]).T
metrics_df
"""
    ),
    nbf.v4.new_code_cell(
        """cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
cm_df = pd.DataFrame(
    cm,
    index=["true benign", "true malignant"],
    columns=["predicted benign", "predicted malignant"],
)

cm_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.xticks([0, 1], ["pred benign", "pred malignant"])
plt.yticks([0, 1], ["true benign", "true malignant"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.title("Confusion Matrix: Linear SVM")
plt.colorbar()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The confusion matrix separates the two main error types:

- False positive: a benign sample predicted as malignant.
- False negative: a malignant sample predicted as benign.

In a screening-oriented health example, false negatives are especially important because they represent missed malignant cases. This is why recall for the malignant class should be interpreted alongside accuracy.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 8. Decision Score Analysis

Unlike logistic regression, this simple linear SVM implementation does not produce calibrated probabilities.

Instead, it produces decision scores:

- Larger positive scores indicate stronger classification as malignant.
- Larger negative scores indicate stronger classification as benign.
- Scores near zero are closer to the decision boundary and are more uncertain.
"""
    ),
    nbf.v4.new_code_cell(
        """score_df = pd.DataFrame(
    {
        "decision_score": decision_scores,
        "true_label": y_test,
        "predicted_label": y_pred,
    }
)

score_df["true_label_name"] = score_df["true_label"].map({0: "benign", 1: "malignant"})
score_df["correct"] = score_df["true_label"] == score_df["predicted_label"]

score_df.head()
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(7, 4))
plt.hist(
    score_df.loc[score_df["true_label"] == 0, "decision_score"],
    bins=25,
    alpha=0.7,
    label="true benign",
)
plt.hist(
    score_df.loc[score_df["true_label"] == 1, "decision_score"],
    bins=25,
    alpha=0.7,
    label="true malignant",
)
plt.axvline(0, linestyle="--", label="decision boundary")
plt.xlabel("Decision score")
plt.ylabel("Frequency")
plt.title("Linear SVM Decision Scores by True Class")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The decision score distribution shows how far samples are from the separating hyperplane. Samples near zero are closer to the decision boundary and are more likely to be ambiguous.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 9. Regularization Sensitivity

The regularization parameter affects the learned margin and coefficient size.

Smaller `alpha` values allow larger coefficients and a more flexible separating boundary. Larger `alpha` values shrink coefficients more strongly and may underfit if the penalty is too strong.
"""
    ),
    nbf.v4.new_code_cell(
        """alpha_values = [0.001, 0.01, 0.1, 1.0]
alpha_results = []

for alpha in alpha_values:
    svm = LinearSVM(
        learning_rate=0.01,
        max_iter=10000,
        alpha=alpha,
        fit_intercept=True,
        tol=1e-8,
    )
    svm.fit(X_train_scaled, y_train)
    pred = svm.predict(X_test_scaled)
    cm_alpha = confusion_matrix(y_test, pred, labels=[0, 1])
    
    alpha_results.append(
        {
            "alpha": alpha,
            "accuracy": accuracy_score(y_test, pred),
            "precision_malignant": precision_score(y_test, pred, pos_label=1),
            "recall_malignant": recall_score(y_test, pred, pos_label=1),
            "f1_malignant": f1_score(y_test, pred, pos_label=1),
            "false_positives": cm_alpha[0, 1],
            "false_negatives": cm_alpha[1, 0],
            "coef_norm": np.linalg.norm(svm.coef_),
            "n_iter": svm.n_iter_,
        }
    )

alpha_results_df = pd.DataFrame(alpha_results)
alpha_results_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.plot(alpha_results_df["alpha"], alpha_results_df["accuracy"], marker="o", label="accuracy")
plt.plot(alpha_results_df["alpha"], alpha_results_df["recall_malignant"], marker="o", label="recall")
plt.plot(alpha_results_df["alpha"], alpha_results_df["f1_malignant"], marker="o", label="F1")
plt.xscale("log")
plt.xlabel("Regularization strength alpha")
plt.ylabel("Metric value")
plt.title("Linear SVM Performance Across Alpha Values")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.plot(alpha_results_df["alpha"], alpha_results_df["coef_norm"], marker="o")
plt.xscale("log")
plt.xlabel("Regularization strength alpha")
plt.ylabel("Coefficient L2 norm")
plt.title("Coefficient Norm Across Alpha Values")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The regularization sensitivity analysis shows how `alpha` changes the trade-off between margin-based fitting and coefficient shrinkage.

If the metrics are similar across several `alpha` values, that suggests the dataset is relatively easy for a linear margin-based classifier under this train/test split. If performance drops at large `alpha`, that suggests excessive regularization can underfit the data.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 10. Coefficient Inspection

Because this is a linear model, we can inspect the learned coefficients.

The coefficients are based on standardized features, so their magnitudes are more comparable than they would be on raw features.

Positive coefficients increase the decision score toward the malignant class. Negative coefficients decrease the score toward the benign class.
"""
    ),
    nbf.v4.new_code_cell(
        """coef_df = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": model.coef_,
        "abs_coefficient": np.abs(model.coef_),
    }
).sort_values("abs_coefficient", ascending=False)

coef_df.head(12)
"""
    ),
    nbf.v4.new_code_cell(
        """top_coef = coef_df.head(12).sort_values("coefficient")

plt.figure(figsize=(8, 5))
plt.barh(top_coef["feature"], top_coef["coefficient"])
plt.axvline(0, linewidth=1)
plt.xlabel("Coefficient")
plt.title("Largest Linear SVM Coefficients")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """Coefficient inspection helps explain the model's decision rule. However, coefficients should be interpreted cautiously because many breast cancer features are correlated. A large coefficient reflects model behavior, not causal importance.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 11. Linear SVM vs. Kernel SVM

This implementation is a linear SVM. It learns a linear decision boundary in the input feature space.

A kernel SVM can learn nonlinear boundaries by implicitly mapping data into a higher-dimensional feature space. Kernel SVMs are more flexible but require additional design choices such as the kernel type and kernel parameters.

For this dataset, a linear SVM is a reasonable baseline because the features are already informative and show strong class separation.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 12. Limitations

This analysis has several limitations:

1. The dataset is a classic benchmark dataset and may be easier than external clinical data.
2. The model is linear and may miss nonlinear structure.
3. This implementation does not include kernel functions.
4. This implementation does not provide calibrated probabilities.
5. Hyperparameters were explored only over a small grid.
6. The result is based on one train/test split rather than repeated cross-validation.
7. Coefficient interpretation is affected by correlated features.
8. This is an educational example, not a clinical diagnostic system.

A stronger analysis could include repeated train/test splits, cross-validation, calibration, external validation, and comparison with kernel SVMs or other nonlinear models.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 13. Conclusion

This notebook demonstrated a linear support vector machine using the custom `LinearSVM` implementation from `jiayi_ml`.

Key takeaways:

- Linear SVM is a margin-based binary classifier.
- Hinge loss penalizes samples that are misclassified or too close to the margin.
- Standardization is important for SVMs.
- Decision scores indicate distance-like confidence relative to the separating hyperplane.
- Regularization affects coefficient size and model flexibility.
- Linear SVM is useful as an interpretable margin-based baseline, but it is less flexible than kernel SVM.
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