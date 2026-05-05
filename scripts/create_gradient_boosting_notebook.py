from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "examples"
    / "supervised_learning"
    / "12_gradient_boosting_regression_diabetes.ipynb"
)

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# Gradient Boosting Regression on the Diabetes Dataset

This notebook demonstrates the `GradientBoostingRegressor` implemented in the `jiayi_ml` package.

The goal is to predict diabetes disease progression using baseline clinical measurements.

This example emphasizes:

1. Supervised regression.
2. Additive ensemble learning.
3. Residual fitting.
4. Squared-error gradient boosting.
5. Learning rate effects.
6. Number of estimators effects.
7. Training loss interpretation.
8. Comparison with simpler regression baselines.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Problem Statement

The Diabetes dataset is a supervised regression dataset.

Each observation contains baseline clinical measurements for one patient, and the target is a quantitative measure of disease progression one year after baseline.

The prediction task is:

> Given baseline clinical measurements, predict the continuous disease progression score.

Gradient boosting is appropriate for this example because it builds an additive model sequentially. Each new tree is fit to the residual errors left by the previous ensemble.

This notebook is an educational machine learning example. It should not be interpreted as a clinical prediction tool.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## Modeling Hypothesis

Because the Diabetes dataset has a continuous outcome and potentially nonlinear relationships among clinical measurements, gradient boosting may improve over a simple linear baseline by fitting residual structure in stages.

The main hypothesis is:

> A gradient boosting regressor will reduce training error as more trees are added, and it may improve test performance if the residual patterns contain useful nonlinear signal.

However, because the dataset is small and the outcome is noisy, gradient boosting may also overfit if the ensemble becomes too complex. Therefore, test-set performance should be interpreted together with the training loss curve and hyperparameter sensitivity.
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
from jiayi_ml.supervised import (
    GradientBoostingRegressor,
    LinearRegression,
    RidgeRegression,
    DecisionTreeRegressor,
)
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

The dataset is loaded from `sklearn.datasets.load_diabetes`, so the notebook is reproducible without external files.

The dataset contains 10 numeric baseline variables and a continuous disease progression target.
"""
    ),
    nbf.v4.new_code_cell(
        """data = load_diabetes(as_frame=True)

X = data.data
y = data.target
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

Before fitting gradient boosting, we inspect:

- Missing values
- Feature summary statistics
- Target distribution
- Feature-target correlations

Gradient boosting can model nonlinear and threshold-like structure, but exploratory analysis is still useful for understanding the prediction problem.
"""
    ),
    nbf.v4.new_code_cell(
        """missing_values = df.isna().sum()

print("Total missing values:", int(missing_values.sum()))

summary = df.describe().T
summary
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(7, 4))
plt.hist(y, bins=25, edgecolor="k", alpha=0.8)
plt.xlabel("Disease progression target")
plt.ylabel("Frequency")
plt.title("Target Distribution")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_code_cell(
        """correlations = df.corr(numeric_only=True)["target"].drop("target")
correlations_sorted = correlations.reindex(
    correlations.abs().sort_values(ascending=False).index
)

correlations_df = correlations_sorted.to_frame(name="correlation_with_target")
correlations_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.bar(correlations_sorted.index, correlations_sorted.values)
plt.axhline(0, linewidth=1)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Correlation with target")
plt.title("Feature Correlations with Disease Progression")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The correlations show which individual features have the strongest linear association with disease progression. Gradient boosting is not limited to purely linear associations, but these correlations provide a useful baseline understanding of the data.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Train/Test Split and Preprocessing

The data is split into training and test sets.

Tree-based models do not require feature standardization in the same way that linear models or distance-based models do. However, this notebook also fits linear baselines, so standardized features are used for the baseline models.

The gradient boosting model is fit on the original feature scale. This is acceptable because regression trees split features using threshold rules.
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

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. Fit Gradient Boosting Regressor

The custom `GradientBoostingRegressor` starts with a constant prediction equal to the training target mean.

Then it repeatedly:

1. Computes residuals.
2. Fits a shallow regression tree to the residuals.
3. Adds the tree prediction to the ensemble after multiplying by the learning rate.

This implementation uses squared-error loss.
"""
    ),
    nbf.v4.new_code_cell(
        """gb_model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=3,
)

gb_model.fit(X_train, y_train)

print("Initial prediction:", gb_model.init_)
print("Number of fitted estimators:", len(gb_model.estimators_))
print("Final training MSE:", gb_model.train_loss_[-1])
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(gb_model.train_loss_) + 1), gb_model.train_loss_, marker="o")
plt.xlabel("Boosting stage")
plt.ylabel("Training MSE")
plt.title("Gradient Boosting Training Loss")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The training loss should generally decrease as more trees are added, because each tree is fit to the remaining residual error. A decreasing training loss confirms that the ensemble is fitting the training data progressively.

However, lower training loss does not automatically imply better generalization. Test-set performance must be evaluated separately.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 6. Test-Set Evaluation

The model is evaluated on held-out test data using:

- Mean squared error
- Root mean squared error
- Mean absolute error
- R-squared

R-squared is useful for interpretation, but it should be treated cautiously on small datasets.
"""
    ),
    nbf.v4.new_code_cell(
        """def regression_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


gb_train_pred = gb_model.predict(X_train)
gb_test_pred = gb_model.predict(X_test)

gb_results = pd.DataFrame(
    {
        "train": regression_metrics(y_train, gb_train_pred),
        "test": regression_metrics(y_test, gb_test_pred),
    }
)

gb_results
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(6, 5))
plt.scatter(y_test, gb_test_pred, alpha=0.8, edgecolors="k")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    label="perfect prediction",
)
plt.xlabel("Actual target")
plt.ylabel("Predicted target")
plt.title("Gradient Boosting: Actual vs. Predicted")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The actual-versus-predicted plot shows how closely the gradient boosting predictions match the held-out target values. Points close to the dashed diagonal line indicate better predictions.

Because the Diabetes dataset is small and noisy, substantial prediction error is expected even for flexible models.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 7. Comparison with Baseline Models

Gradient boosting is compared with:

- Linear regression
- Ridge regression
- A single decision tree regressor

The linear models use standardized features. The decision tree and gradient boosting models use the original feature scale.
"""
    ),
    nbf.v4.new_code_cell(
        """baseline_models = {
    "Linear Regression": (LinearRegression(), X_train_scaled, X_test_scaled),
    "Ridge Regression": (RidgeRegression(alpha=1.0), X_train_scaled, X_test_scaled),
    "Decision Tree": (
        DecisionTreeRegressor(max_depth=3, min_samples_split=2),
        X_train,
        X_test,
    ),
    "Gradient Boosting": (gb_model, X_train, X_test),
}

comparison_rows = []

for model_name, (model, Xtr, Xte) in baseline_models.items():
    if model_name != "Gradient Boosting":
        model.fit(Xtr, y_train)
    
    train_pred = model.predict(Xtr)
    test_pred = model.predict(Xte)
    
    row = {
        "model": model_name,
        "train_RMSE": root_mean_squared_error(y_train, train_pred),
        "test_RMSE": root_mean_squared_error(y_test, test_pred),
        "train_R2": r2_score(y_train, train_pred),
        "test_R2": r2_score(y_test, test_pred),
    }
    comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows).sort_values("test_RMSE")
comparison_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.bar(comparison_df["model"], comparison_df["test_RMSE"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("Test RMSE")
plt.title("Test RMSE Comparison")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The comparison table should be interpreted empirically rather than assumed in advance. Gradient boosting often improves over a single tree by combining many weak learners, but it does not always outperform simpler linear models on small tabular datasets.

If gradient boosting has lower training error but not lower test error, that suggests the model may be fitting training residuals without improving generalization.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 8. Number of Estimators Sensitivity

The number of boosting stages controls ensemble complexity.

Too few trees may underfit because the ensemble cannot capture enough residual structure. Too many trees may overfit, especially when the learning rate is not small enough.
"""
    ),
    nbf.v4.new_code_cell(
        """estimator_values = [1, 5, 10, 25, 50, 100]
estimator_results = []

for n_estimators in estimator_values:
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=3,
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    estimator_results.append(
        {
            "n_estimators": n_estimators,
            "train_RMSE": root_mean_squared_error(y_train, train_pred),
            "test_RMSE": root_mean_squared_error(y_test, test_pred),
            "train_R2": r2_score(y_train, train_pred),
            "test_R2": r2_score(y_test, test_pred),
        }
    )

estimator_results_df = pd.DataFrame(estimator_results)
estimator_results_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.plot(
    estimator_results_df["n_estimators"],
    estimator_results_df["train_RMSE"],
    marker="o",
    label="train RMSE",
)
plt.plot(
    estimator_results_df["n_estimators"],
    estimator_results_df["test_RMSE"],
    marker="o",
    label="test RMSE",
)
plt.xlabel("Number of estimators")
plt.ylabel("RMSE")
plt.title("Effect of Number of Estimators")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The estimator sensitivity table and plot show whether adding more trees improves generalization or mainly reduces training error.

A widening train-test gap would suggest overfitting. A stable or decreasing test RMSE suggests that additional boosting stages are still useful under the chosen learning rate and tree depth.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 9. Learning Rate Sensitivity

The learning rate controls how much each tree contributes to the ensemble.

Smaller learning rates usually require more trees but can produce smoother, more stable learning. Larger learning rates fit faster but may overfit or become less stable.
"""
    ),
    nbf.v4.new_code_cell(
        """learning_rates = [0.01, 0.05, 0.1, 0.2]
learning_rate_results = []

for learning_rate in learning_rates:
    model = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=learning_rate,
        max_depth=2,
        min_samples_leaf=3,
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    learning_rate_results.append(
        {
            "learning_rate": learning_rate,
            "train_RMSE": root_mean_squared_error(y_train, train_pred),
            "test_RMSE": root_mean_squared_error(y_test, test_pred),
            "train_R2": r2_score(y_train, train_pred),
            "test_R2": r2_score(y_test, test_pred),
        }
    )

learning_rate_results_df = pd.DataFrame(learning_rate_results)
learning_rate_results_df
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.plot(
    learning_rate_results_df["learning_rate"],
    learning_rate_results_df["train_RMSE"],
    marker="o",
    label="train RMSE",
)
plt.plot(
    learning_rate_results_df["learning_rate"],
    learning_rate_results_df["test_RMSE"],
    marker="o",
    label="test RMSE",
)
plt.xlabel("Learning rate")
plt.ylabel("RMSE")
plt.title("Effect of Learning Rate")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The learning rate sensitivity analysis illustrates the shrinkage effect in gradient boosting. A smaller learning rate may underfit if the number of trees is limited, while a larger learning rate can reduce training error more aggressively.

The best learning rate should be selected using validation or cross-validation rather than the test set.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 10. Staged Test Performance

The `staged_predict` method allows us to inspect how test performance changes after each boosting stage.

This helps diagnose whether adding more trees continues to improve generalization or mainly improves training fit.
"""
    ),
    nbf.v4.new_code_cell(
        """stage_rows = []

for stage_index, staged_pred in enumerate(gb_model.staged_predict(X_test), start=1):
    stage_rows.append(
        {
            "stage": stage_index,
            "test_RMSE": root_mean_squared_error(y_test, staged_pred),
            "test_R2": r2_score(y_test, staged_pred),
        }
    )

stage_df = pd.DataFrame(stage_rows)
stage_df.head()
"""
    ),
    nbf.v4.new_code_cell(
        """plt.figure(figsize=(8, 4))
plt.plot(stage_df["stage"], stage_df["test_RMSE"])
plt.xlabel("Boosting stage")
plt.ylabel("Test RMSE")
plt.title("Staged Test RMSE")
plt.tight_layout()
plt.show()
"""
    ),
    nbf.v4.new_markdown_cell(
        """The staged test curve is useful because gradient boosting is sequential. If test RMSE decreases and then increases, that may indicate overfitting after too many stages.

This notebook uses a simple train/test split, so the curve should be interpreted as exploratory rather than as a final model-selection procedure.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 11. Interpretation

Gradient boosting reduces training error by fitting a sequence of shallow trees to residuals. This can capture nonlinear patterns that linear models may miss.

However, on small biomedical-style tabular datasets, a more flexible model does not automatically guarantee better test performance. If the linear baseline performs similarly or better, that suggests the dataset may not contain enough stable nonlinear signal for gradient boosting to exploit under this split.

The main value of this notebook is to show how gradient boosting learns sequentially and how hyperparameters such as number of estimators and learning rate affect the bias-variance trade-off.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 12. Limitations

This analysis has several limitations:

1. The dataset is small.
2. The result is based on one train/test split.
3. Hyperparameter tuning is limited.
4. The custom implementation uses squared-error regression only.
5. The custom implementation is not optimized for large datasets.
6. The test set is used for demonstration, not for formal model selection.
7. Clinical generalization would require external validation.

A stronger analysis could use cross-validation, validation curves, early stopping, and comparison with optimized library implementations.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 13. Conclusion

This notebook demonstrated gradient boosting regression using the custom `GradientBoostingRegressor` implementation from `jiayi_ml`.

Key takeaways:

- Gradient boosting builds an additive ensemble sequentially.
- Each tree is fit to the residuals of the current ensemble.
- Training loss generally decreases as more trees are added.
- Learning rate and number of estimators control model complexity.
- Gradient boosting can model nonlinear structure, but it may overfit small datasets.
- Test-set evaluation is necessary because training loss alone does not measure generalization.
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