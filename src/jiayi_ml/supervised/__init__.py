"""
Supervised learning algorithms for jiayi_ml.
"""

from jiayi_ml.supervised.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from jiayi_ml.supervised.knn import KNNClassifier, KNNRegressor
from jiayi_ml.supervised.linear_regression import (
    LassoRegression,
    LinearRegression,
    RidgeRegression,
)
from jiayi_ml.supervised.logistic_regression import LogisticRegression
from jiayi_ml.supervised.naive_bayes import GaussianNaiveBayes
from jiayi_ml.supervised.perceptron import Perceptron
from jiayi_ml.supervised.random_forest import (
    RandomForestClassifier,
    RandomForestRegressor,
)

__all__ = [
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "LogisticRegression",
    "KNNClassifier",
    "KNNRegressor",
    "GaussianNaiveBayes",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "Perceptron",
]