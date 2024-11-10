
"""
List of available models.
"""

from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression import LassoRegression
from autoop.core.ml.model.regression import RidgeRegression
from autoop.core.ml.model.classification import LogisticRegressor
from autoop.core.ml.model.classification import KNearestNeighbors
from autoop.core.ml.model.classification import SupportVectorClassifier
from autoop.core.ml.model.base_model import Model


REGRESSION_MODELS = [
    "Lasso Regression",
    "Multiple Linear Regression",
    "Ridge Regression"
]

CLASSIFICATION_MODELS = [
    "Logistic Regressor",
    "KNN",
    "Support Vector Classifier"
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name == "Multiple Linear Regression":
        return MultipleLinearRegression()
    elif model_name == "Lasso Regression":
        return LassoRegression()
    elif model_name == "Ridge Regression":
        return RidgeRegression()
    elif model_name == "Logistic Regressor":
        return LogisticRegressor()
    elif model_name == "KNN":
        return KNearestNeighbors()
    elif model_name == "Support Vector Classifier":
        return SupportVectorClassifier()
