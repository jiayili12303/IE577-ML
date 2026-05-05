from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "examples"
    / "supervised_learning"
    / "02_logistic_regression_breast_cancer.ipynb"
)

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Logistic Regression on the Breast Cancer Dataset

This notebook demonstrates the `LogisticRegression` classifier implemented in the `jiayi_ml` package.

The goal is to classify breast tumor samples as malignant or benign using numeric features computed from digitized images of breast mass cell nuclei.

This example emphasizes a complete binary classification workflow:

1. Define the classification problem.
2. Load and inspect the dataset.
3. Perform exploratory data analysis.
4. Recode the target so malignant tumors are treated as the positive class.
5. Split the data into training and test sets.
6. Apply leakage-safe standardization.
7. Fit the custom logistic regression model.
8. Evaluate the model using accuracy, precision, recall, F1 score, and confusion matrix.
9. Analyze decision thresholds.
10. Discuss false positives, false negatives, and limitations.
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

That choice is important for interpretation. In a health-related screening example, a false negative means a malignant tumor is incorrectly predicted as benign. This type of error is usually more concerning than a false positive because it may delay follow-up evaluation.

This notebook is an educational machine learning example. It should not be interpreted as a clinical diagnostic tool.
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
from jiayi_ml.supervised import LogisticRegression
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

The dataset is loaded from `sklearn.datasets.load_breast_cancer`, so this notebook is fully reproducible and does not require external files.

The features are numeric measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, and related summary statistics.

The original sklearn target encoding is:

- `0` = malignant
- `1` = benign

For evaluation, this notebook recodes the target so that malignant is the positive class.
"""
    ),
    nbf.v4.new_code_cell(
        """data = load_breast_cancer(as_frame=True)

X = data.data
y_original = data.target
target_names = data.target_names
feature_names = X.columns.tolist()

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
        """## 3. Exploratory Data Analysis

The first step is to inspect the dataset size, missing values, class balance, and basic feature summaries.

Class balance is especially important for binary classification because accuracy can be misleading if one class is much more common than the other.
"""
    ),
    nbf.v4.new_code_cell(
        """missing_values = df.isna().sum()
print("Total missing values:", int(missing_values.sum()))

summary = df.describe().T
summary.head(10)
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
plt.bar(class_counts.index, class_counts.values)
plt.ylabel("Number of samples")
plt.title("Class Distribution")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The dataset contains both benign and malignant samples, but the classes are not perfectly balanced. This motivates using metrics beyond accuracy, especially recall for the malignant class.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Feature Exploration

The dataset contains 30 numeric features. Many are related measurements computed from the same underlying cell nuclei characteristics, so strong correlations between features are expected.

The following analysis looks at which features have the strongest marginal correlation with the malignant indicator. This is not a causal analysis, but it helps identify features that may be informative for classification.
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
        """Features with larger absolute correlations may help separate malignant and benign tumors in a linear classifier. Logistic regression is a natural baseline because it models the log-odds of the positive class as a linear function of the features.

However, many biomedical measurements are correlated. That means coefficient interpretation should be cautious: a large coefficient does not prove that a feature independently causes malignancy.
"""
    ),
    nbf.v4.new_code_cell(
        """selected_features = top_correlations.index[:4].tolist()

for feature in selected_features:
    plt.figure(figsize=(6, 4))
    plt.hist(X.loc[y == 0, feature], bins=25, alpha=0.7, label="benign")
    plt.hist(X.loc[y == 1, feature], bins=25, alpha=0.7, label="malignant")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature} by Class")
    plt.legend()
    plt.tight_layout()
    plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The feature distributions show that some measurements differ substantially between benign and malignant samples, although overlap remains. This overlap is why classification errors are possible and why confusion matrix analysis is important.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. Train/Test Split and Preprocessing

The data is split into training and test sets using stratification so that both sets preserve the benign/malignant class distribution.

Standardization is fit only on the training data and then applied to the test data. This avoids data leakage.
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
        """## 6. Fit Custom Logistic Regression

The model below uses the `LogisticRegression` class from the local `jiayi_ml` package.

L2 regularization is included to reduce coefficient instability because many features in this dataset are correlated.
"""
    ),
    nbf.v4.new_code_cell(
        """model = LogisticRegression(
    learning_rate=0.05,
    max_iter=10000,
    penalty="l2",
    alpha=0.01,
    tol=1e-8,
)

model.fit(X_train_scaled, y_train)

print("Number of iterations:", model.n_iter_)
print("Final training loss:", model.losses_[-1])
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(7, 4))
plt.plot(model.losses_)
plt.xlabel("Iteration")
plt.ylabel("Binary cross-entropy loss")
plt.title("Training Loss Curve")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The decreasing loss curve provides a basic check that gradient descent is optimizing the logistic regression objective.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 7. Test-Set Evaluation at Threshold 0.50

By default, logistic regression predicts the positive class when the predicted probability is at least 0.50.

Because malignant is encoded as the positive class, the positive-class probability is the model's estimated probability of malignancy.
"""
    ),
    nbf.v4.new_code_cell(
        """y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled, threshold=0.5)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, pos_label=1),
    "recall": recall_score(y_test, y_pred, pos_label=1),
    "f1": f1_score(y_test, y_pred, pos_label=1),
}

metrics_df = pd.DataFrame(metrics, index=["threshold_0.50"]).T
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

plt.title("Confusion Matrix at Threshold 0.50")
plt.colorbar()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The confusion matrix separates the types of errors:

- False positive: a benign case predicted as malignant.
- False negative: a malignant case predicted as benign.

In a screening-oriented health example, false negatives are particularly important because they represent missed malignant cases. Precision and recall should therefore be interpreted alongside accuracy.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 8. Threshold Analysis

The classification threshold controls the trade-off between precision and recall.

A lower threshold predicts malignant more often. This can increase recall but may also increase false positives.

A higher threshold predicts malignant only when the model is more confident. This can improve precision but may reduce recall.
"""
    ),
    nbf.v4.new_code_cell(
        """def evaluate_threshold(threshold):
    predictions = np.where(y_proba >= threshold, 1, 0)
    cm_threshold = confusion_matrix(y_test, predictions, labels=[0, 1])
    
    false_positives = cm_threshold[0, 1]
    false_negatives = cm_threshold[1, 0]
    
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label=1),
        "recall": recall_score(y_test, predictions, pos_label=1),
        "f1": f1_score(y_test, predictions, pos_label=1),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
threshold_results = pd.DataFrame([evaluate_threshold(t) for t in thresholds])
threshold_results
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.plot(threshold_results["threshold"], threshold_results["precision"], marker="o", label="precision")
plt.plot(threshold_results["threshold"], threshold_results["recall"], marker="o", label="recall")
plt.plot(threshold_results["threshold"], threshold_results["f1"], marker="o", label="f1")
plt.xlabel("Decision threshold")
plt.ylabel("Metric value")
plt.title("Precision, Recall, and F1 Across Thresholds")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.plot(
    threshold_results["threshold"],
    threshold_results["false_positives"],
    marker="o",
    label="false positives",
)
plt.plot(
    threshold_results["threshold"],
    threshold_results["false_negatives"],
    marker="o",
    label="false negatives",
)
plt.xlabel("Decision threshold")
plt.ylabel("Number of errors")
plt.title("False Positives and False Negatives Across Thresholds")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The threshold analysis shows that classification is not only about fitting a model. It also requires choosing an operating point.

If the goal is to reduce missed malignant cases, a lower threshold may be preferred because it tends to increase recall. However, this usually comes with more false positives. In practice, the threshold should depend on the cost of different errors and the intended use of the model.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 9. Coefficient Inspection

Because logistic regression is a linear model, the learned coefficients can be inspected.

The coefficients are shown on standardized features, which makes their magnitudes more comparable. Positive coefficients increase the log-odds of the malignant class, while negative coefficients decrease it.

This interpretation is associational and model-based. It should not be interpreted as causal.
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
plt.title("Largest Logistic Regression Coefficients")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The largest coefficients indicate features that contribute most strongly to the model's linear decision rule after standardization. However, because many features are correlated, coefficient magnitude should be interpreted as model behavior rather than as independent biological importance.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 10. Limitations

This notebook has several limitations:

1. The dataset is relatively small.
2. The features are derived from image measurements, but this notebook only models tabular numeric summaries.
3. Logistic regression assumes a linear decision boundary in the feature space.
4. Feature correlations can make coefficient interpretation unstable.
5. Threshold selection should depend on the intended use case and error costs.
6. This is an educational example and not a clinical diagnostic system.

A stronger analysis could include cross-validation, calibration analysis, external validation, and comparison with additional models.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 11. Conclusion

This notebook demonstrated binary classification using the custom `LogisticRegression` implementation from `jiayi_ml`.

Key takeaways:

- Logistic regression is an interpretable baseline for binary classification.
- Accuracy alone is not enough for health-related classification examples.
- Confusion matrices help separate false positives from false negatives.
- Recall is especially important when missed positive cases are costly.
- Decision thresholds control the trade-off between precision and recall.
- Coefficients can help explain model behavior, but they should be interpreted cautiously.
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