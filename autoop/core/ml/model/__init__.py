
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression

REGRESSION_MODELS = [
    "lasso",
    "multiple_linear_regression",
    "ridge_regression"
]

CLASSIFICATION_MODELS = [
    "logistic_regression",
    "knn",
    "support_vector_machines"
]

def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    raise NotImplementedError("To be implemented.")