from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "examples"
    / "supervised_learning"
    / "01_linear_ridge_lasso_regression_diabetes.ipynb"
)

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Linear, Ridge, and Lasso Regression on the Diabetes Dataset

This notebook demonstrates three regression algorithms implemented in the `jiayi_ml` package:

- `LinearRegression`
- `RidgeRegression`
- `LassoRegression`

The goal is to predict a continuous diabetes disease progression score using baseline clinical measurements.

This example emphasizes a complete supervised learning workflow:

1. Define the regression problem.
2. Load and inspect the dataset.
3. Perform exploratory data analysis.
4. Split the data into training and test sets.
5. Apply preprocessing without data leakage.
6. Fit custom regression models.
7. Evaluate performance on held-out test data.
8. Compare coefficient behavior across linear, ridge, and lasso regression.
9. Discuss strengths, limitations, and interpretation.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Problem Statement

The Diabetes dataset contains baseline clinical measurements for patients, along with a continuous target measuring disease progression one year after baseline.

The prediction task is:

> Given patient baseline measurements, predict the continuous disease progression score.

This is a **supervised regression** problem because:

- The target variable is known during training.
- The target is continuous.
- The goal is numerical prediction rather than classification or clustering.

The three models in this notebook represent different assumptions about linear prediction:

- **Linear Regression** fits an ordinary least squares model.
- **Ridge Regression** adds an L2 penalty, shrinking coefficients smoothly.
- **Lasso Regression** adds an L1 penalty, encouraging sparse coefficients.
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

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from jiayi_ml.preprocessing import StandardScaler
from jiayi_ml.supervised import LinearRegression, RidgeRegression, LassoRegression
from jiayi_ml.metrics import (
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
        """## 2. Load the Dataset

The dataset is loaded from `sklearn.datasets.load_diabetes`, which makes the notebook fully reproducible without requiring external downloads.

The feature columns include baseline clinical measurements such as age, sex, body mass index, blood pressure, and several serum measurements.
"""
    ),
    nbf.v4.new_code_cell(
        """diabetes = load_diabetes(as_frame=True)

X = diabetes.data
y = diabetes.target
feature_names = X.columns.tolist()

df = X.copy()
df["target"] = y

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("Feature names:", feature_names)

df.head()
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 3. Exploratory Data Analysis

Before modeling, it is important to understand the dataset structure.

The following checks focus on:

- Missing values
- Target distribution
- Feature summary statistics
- Correlation between features and the target
"""
    ),
    nbf.v4.new_code_cell(
        """missing_values = df.isna().sum()

summary = df.describe().T

print("Missing values by column:")
print(missing_values)

summary
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(7, 4))
plt.hist(y, bins=25)
plt.xlabel("Disease progression score")
plt.ylabel("Frequency")
plt.title("Distribution of Diabetes Disease Progression Target")
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The target is continuous and moderately spread out, which supports treating this as a regression problem. A model should be evaluated by how close its predictions are to the true disease progression scores, not by classification accuracy.
"""
    ),
    nbf.v4.new_code_cell(
        """correlations = df.corr(numeric_only=True)["target"].drop("target").sort_values(
    key=lambda values: values.abs(),
    ascending=False,
)

correlations_df = correlations.to_frame(name="correlation_with_target")
correlations_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.bar(correlations.index, correlations.values)
plt.axhline(0, linewidth=1)
plt.xticks(rotation=45)
plt.ylabel("Correlation with target")
plt.title("Feature Correlations with Disease Progression")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The exploratory correlation analysis gives a first look at which features have stronger linear relationships with the target. This does not prove causality, but it helps motivate why linear models may be reasonable baselines for this dataset.

Because ridge and lasso regression operate directly on coefficient magnitudes, standardization is important. Even though this sklearn dataset is already numerically scaled, the notebook still demonstrates a leakage-safe preprocessing workflow.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Train/Test Split and Preprocessing

The dataset is split into training and test sets.

The scaler is fit only on the training data and then applied to both training and test data. This avoids data leakage because information from the test set is not used to estimate preprocessing parameters.
"""
    ),
    nbf.v4.new_code_cell(
        """X_train, X_test, y_train, y_test = train_test_split(
    X.values,
    y.values,
    test_size=0.25,
    random_state=438,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training set shape:", X_train_scaled.shape)
print("Test set shape:", X_test_scaled.shape)
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. Fit Custom Regression Models

We fit three models implemented in the local `jiayi_ml` package:

1. Ordinary linear regression
2. Ridge regression with L2 regularization
3. Lasso regression with L1 regularization

The regularization strength values below are chosen to demonstrate the difference between coefficient shrinkage patterns. In a larger analysis, these values should be tuned using cross-validation.
"""
    ),
    nbf.v4.new_code_cell(
        """models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": RidgeRegression(alpha=1.0),
    "Lasso Regression": LassoRegression(alpha=0.05, max_iter=10000, tol=1e-8),
}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    print(f"{model_name} fitted.")
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 6. Evaluate on the Held-Out Test Set

For regression, we use the following metrics:

- **MSE**: average squared prediction error.
- **RMSE**: square root of MSE, on the same scale as the target.
- **MAE**: average absolute prediction error.
- **R2**: proportion of variance explained by the model.

Evaluation is performed on the test set to estimate generalization performance.
"""
    ),
    nbf.v4.new_code_cell(
        """results = []

for model_name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    
    results.append(
        {
            "model": model_name,
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
        }
    )

results_df = pd.DataFrame(results).set_index("model")
results_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(7, 4))
plt.bar(results_df.index, results_df["RMSE"])
plt.ylabel("Test RMSE")
plt.title("Test RMSE Comparison Across Regression Models")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """Lower RMSE indicates smaller average prediction error. Ridge and lasso may improve test performance if the ordinary linear regression model is unstable or overfits noisy features. However, regularization can also hurt performance if the penalty is too strong.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 7. Predicted vs. True Values

A predicted-versus-true plot helps assess whether model predictions track the target values across the range of disease progression scores.

A perfect model would place all points on the diagonal line.
"""
    ),
    nbf.v4.new_code_cell(
        """for model_name, model in models.items():
    y_pred = model.predict(X_test_scaled)

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    
    lower = min(np.min(y_test), np.min(y_pred))
    upper = max(np.max(y_test), np.max(y_pred))
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    
    plt.xlabel("True disease progression score")
    plt.ylabel("Predicted disease progression score")
    plt.title(f"Predicted vs. True Values: {model_name}")
    plt.tight_layout()
    plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 8. Coefficient Comparison

Linear, ridge, and lasso regression differ mainly in how they estimate coefficients.

- Linear regression has no penalty.
- Ridge regression shrinks coefficients toward zero but usually keeps all features.
- Lasso regression can shrink some coefficients exactly to zero, creating a sparse model.

This is useful when we want a simpler model or want to identify a smaller subset of influential predictors.
"""
    ),
    nbf.v4.new_code_cell(
        """coef_df = pd.DataFrame(
    {
        model_name: model.coef_
        for model_name, model in models.items()
    },
    index=feature_names,
)

coef_df
"""
    ),
    nbf.v4.new_code_cell(
        """coef_df.plot(kind="bar", figsize=(10, 5))
plt.axhline(0, linewidth=1)
plt.ylabel("Coefficient value")
plt.title("Coefficient Comparison Across Linear, Ridge, and Lasso Regression")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """lasso_nonzero = coef_df["Lasso Regression"].abs() > 1e-8

print("Number of nonzero Lasso coefficients:", int(lasso_nonzero.sum()))
print("Features retained by Lasso:")
print(coef_df.index[lasso_nonzero].tolist())
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 9. Interpretation

The coefficient comparison illustrates the effect of regularization.

Ridge regression typically reduces coefficient magnitude while keeping most variables in the model. This can improve stability when predictors are correlated.

Lasso regression can produce a sparse coefficient vector. In applied biomedical modeling, sparsity can be useful because it produces a simpler model and may highlight a smaller set of potentially important predictors. However, feature selection from lasso should not be interpreted as causal evidence.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 10. Limitations

This analysis has several limitations:

1. The dataset is relatively small.
2. The models are linear and may miss nonlinear relationships.
3. The regularization strengths were chosen for demonstration rather than tuned by cross-validation.
4. Correlation and coefficient magnitude do not imply causality.
5. The target is a disease progression score, but this notebook is an educational modeling example rather than a clinical prediction tool.

A stronger analysis could include cross-validation, hyperparameter tuning, nonlinear models, and uncertainty estimates.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 11. Conclusion

This notebook demonstrated a complete regression workflow using custom implementations from `jiayi_ml`.

Key takeaways:

- Linear regression provides a simple baseline for continuous prediction.
- Ridge regression stabilizes coefficients through L2 regularization.
- Lasso regression encourages sparsity through L1 regularization.
- Test-set evaluation is necessary to compare generalization performance.
- Coefficient analysis can help interpret model behavior, but it should be discussed carefully.
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