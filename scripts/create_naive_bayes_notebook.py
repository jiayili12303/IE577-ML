from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "examples"
    / "supervised_learning"
    / "05_naive_bayes_breast_cancer.ipynb"
)

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Gaussian Naive Bayes on the Breast Cancer Dataset

This notebook demonstrates the `GaussianNaiveBayes` classifier implemented in the `jiayi_ml` package.

The goal is to classify breast tumor samples as malignant or benign using numeric features derived from digitized images of breast mass cell nuclei.

This example emphasizes:

1. Probabilistic classification.
2. Class-conditional Gaussian assumptions.
3. The Naive Bayes conditional independence assumption.
4. Binary classification evaluation.
5. Confusion matrix interpretation.
6. False positive and false negative analysis.
7. Limitations of probabilistic assumptions in biomedical-style data.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Problem Statement

The Breast Cancer Wisconsin dataset is a binary classification dataset.

Each observation describes a tumor sample using numeric features computed from cell nuclei. The original target labels are:

- malignant
- benign

For this notebook, the target is recoded so that:

- `1` = malignant
- `0` = benign

This makes malignant tumors the positive class.

The prediction task is:

> Given numeric tumor measurements, predict whether the tumor is malignant or benign.

This is a supervised binary classification task.

Gaussian Naive Bayes is a useful model for this example because it produces class probabilities and has a clear probabilistic interpretation. However, its assumptions are strong, especially the assumption that features are conditionally independent given the class label.

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
from jiayi_ml.supervised import GaussianNaiveBayes
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

The dataset is loaded from `sklearn.datasets.load_breast_cancer`, so this notebook is reproducible without external downloads.

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
        """## 3. Exploratory Data Analysis

Before fitting a probabilistic classifier, we inspect:

- Missing values
- Class distribution
- Feature summaries
- Feature relationships with the target
- Feature correlations

This is important because Gaussian Naive Bayes makes assumptions about feature distributions and feature independence.
"""
    ),
    nbf.v4.new_code_cell(
        """missing_values = df.isna().sum()

print("Total missing values:", int(missing_values.sum()))

summary = X.describe().T
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
plt.bar(class_counts_df.index, class_counts_df["count"])
plt.ylabel("Number of samples")
plt.title("Class Distribution")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The classes are not perfectly balanced. Because malignant is the positive class, evaluation should include precision, recall, F1 score, and a confusion matrix rather than accuracy alone.
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
        """The correlation analysis identifies features that are strongly associated with the malignant class. This does not imply causality, but it helps us understand which variables may be informative for classification.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Checking Distributional and Independence Assumptions

Gaussian Naive Bayes assumes that, within each class, each feature follows a Gaussian distribution. It also assumes that features are conditionally independent given the class.

The following plots inspect selected high-correlation features by class. The distributions do not need to be perfectly Gaussian for the model to be useful, but strong departures from the assumptions should be acknowledged.
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
    plt.title(f"Class-Conditional Distribution: {feature}")
    plt.legend()
    plt.tight_layout()
    plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """# Inspect correlations among the most target-associated features.
corr_features = top_correlations.index[:8].tolist()
corr_matrix = X[corr_features].corr()

plt.figure(figsize=(7, 6))
plt.imshow(corr_matrix, vmin=-1, vmax=1)
plt.xticks(range(len(corr_features)), corr_features, rotation=60, ha="right")
plt.yticks(range(len(corr_features)), corr_features)
plt.colorbar(label="correlation")
plt.title("Feature Correlation Matrix for Selected Features")
plt.tight_layout()
plt.show()

corr_matrix
"""
    ),
    nbf.v4.new_markdown_cell(
        """The correlation matrix shows that many features are strongly correlated. This is important because Gaussian Naive Bayes assumes conditional independence among features given the class.

In biomedical-style tabular data, this assumption is often unrealistic because measurements may be derived from related biological or imaging processes. Even so, Naive Bayes can still perform well as a simple probabilistic baseline.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. Train/Test Split and Preprocessing

The data is split into training and test sets using stratification to preserve the class distribution.

Gaussian Naive Bayes estimates a mean and variance for each feature within each class. Standardization is not strictly required for Gaussian Naive Bayes, but it can improve numerical stability and keeps preprocessing consistent across examples.

The scaler is fit only on the training data and then applied to the test data to avoid data leakage.
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
        """## 6. Fit Custom Gaussian Naive Bayes

We fit the `GaussianNaiveBayes` model from the local `jiayi_ml` package.

The model estimates:

- Class prior probabilities
- Class-specific feature means
- Class-specific feature variances

Predictions are made by choosing the class with the largest posterior probability.
"""
    ),
    nbf.v4.new_code_cell(
        """model = GaussianNaiveBayes(var_smoothing=1e-9)
model.fit(X_train_scaled, y_train)

print("Classes:", model.classes_)
print("Class priors:", model.class_prior_)
print("Theta shape:", model.theta_.shape)
print("Variance shape:", model.var_.shape)
"""
    ),
    nbf.v4.new_markdown_cell(
        """The class priors reflect the class distribution in the training set. The `theta_` matrix contains estimated feature means for each class, and the `var_` matrix contains estimated feature variances for each class.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 7. Test-Set Evaluation

The default prediction rule assigns each sample to the class with the highest posterior probability.

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
y_proba = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision_malignant": precision_score(y_test, y_pred, pos_label=1),
    "recall_malignant": recall_score(y_test, y_pred, pos_label=1),
    "f1_malignant": f1_score(y_test, y_pred, pos_label=1),
}

metrics_df = pd.DataFrame(metrics, index=["Gaussian Naive Bayes"]).T
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

plt.title("Confusion Matrix: Gaussian Naive Bayes")
plt.colorbar()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The confusion matrix separates correct predictions from two different error types:

- False positive: benign predicted as malignant.
- False negative: malignant predicted as benign.

In a screening-oriented health example, false negatives are particularly important because they represent missed malignant cases. This is why recall for the malignant class is important to inspect.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 8. Probability Distribution and Confidence

Naive Bayes produces posterior class probabilities. These probabilities can be inspected to understand how confident the model is.

The following plot shows the predicted probability of malignancy for test samples, separated by true class.
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(7, 4))
plt.hist(y_proba[y_test == 0], bins=20, alpha=0.7, label="true benign")
plt.hist(y_proba[y_test == 1], bins=20, alpha=0.7, label="true malignant")
plt.xlabel("Predicted probability of malignancy")
plt.ylabel("Frequency")
plt.title("Predicted Probability Distribution by True Class")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """Ideally, true malignant cases should have high predicted probabilities, while true benign cases should have low predicted probabilities. Overlap between these distributions indicates uncertainty and potential classification errors.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 9. Threshold Analysis

Although Naive Bayes predicts the class with the highest posterior probability by default, we can still adjust the threshold for predicting the malignant class.

Lowering the threshold usually increases recall but may increase false positives. Raising the threshold may increase precision but can increase false negatives.
"""
    ),
    nbf.v4.new_code_cell(
        """def evaluate_threshold(threshold):
    predictions = np.where(y_proba >= threshold, 1, 0)
    cm_threshold = confusion_matrix(y_test, predictions, labels=[0, 1])
    
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label=1),
        "recall": recall_score(y_test, predictions, pos_label=1),
        "f1": f1_score(y_test, predictions, pos_label=1),
        "false_positives": cm_threshold[0, 1],
        "false_negatives": cm_threshold[1, 0],
    }


thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
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
        """Threshold analysis is important because the default threshold may not match the practical goal of the model.

If the goal is to reduce missed malignant cases, a lower threshold may be preferred because it tends to increase recall. However, this can increase false positives. The final threshold should depend on the relative cost of different error types.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 10. Inspecting Class-Conditional Means

Gaussian Naive Bayes estimates a mean for each feature within each class.

The difference between class means can help identify features that separate the classes under the model. Because the data was standardized, these differences are on a comparable scale.
"""
    ),
    nbf.v4.new_code_cell(
        """mean_difference = model.theta_[1] - model.theta_[0]

mean_diff_df = pd.DataFrame(
    {
        "feature": feature_names,
        "mean_malignant_minus_benign": mean_difference,
        "abs_difference": np.abs(mean_difference),
    }
).sort_values("abs_difference", ascending=False)

mean_diff_df.head(12)
"""
    ),
    nbf.v4.new_code_cell(
        """top_mean_diff = mean_diff_df.head(12).sort_values("mean_malignant_minus_benign")

plt.figure(figsize=(8, 5))
plt.barh(top_mean_diff["feature"], top_mean_diff["mean_malignant_minus_benign"])
plt.axvline(0, linewidth=1)
plt.xlabel("Standardized mean difference")
plt.title("Largest Class-Conditional Mean Differences")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The class-conditional mean differences show which standardized features have the largest average differences between malignant and benign samples.

This is useful for interpretation, but it is not causal evidence. It also does not fully account for correlations among features.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 11. Interpretation

Gaussian Naive Bayes is attractive because it is simple, fast, and probabilistic. It estimates a class-specific Gaussian distribution for each feature and combines the evidence using Bayes' rule.

For this dataset, the model can perform well because several features show strong differences between benign and malignant samples.

However, the correlation analysis shows that the conditional independence assumption is likely imperfect. Many features are related measurements derived from similar image characteristics. This means the model's probability estimates should be interpreted cautiously.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 12. Limitations

This analysis has several limitations:

1. The dataset is relatively small.
2. Gaussian Naive Bayes assumes class-conditional Gaussian feature distributions.
3. The model assumes features are conditionally independent given the class.
4. Many features in this dataset are correlated, which violates the independence assumption.
5. Probability estimates may be poorly calibrated.
6. The threshold should be chosen based on the intended use case.
7. This is an educational example, not a clinical diagnostic system.

A stronger analysis could include calibration curves, cross-validation, external validation, and comparison with other classifiers.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 13. Conclusion

This notebook demonstrated Gaussian Naive Bayes using the custom `GaussianNaiveBayes` implementation from `jiayi_ml`.

Key takeaways:

- Gaussian Naive Bayes is a simple probabilistic classifier.
- It estimates class priors, class-specific feature means, and class-specific feature variances.
- Confusion matrices help identify false positives and false negatives.
- Threshold analysis changes the precision-recall trade-off.
- The independence assumption is important and often unrealistic in biomedical-style data.
- Even when assumptions are imperfect, Naive Bayes can be useful as a transparent baseline.
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