from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression

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


models_list = Model.__iter__
def get_model(model_name: str) -> Model | None:
    """Factory function to get a model by name."""
    for model in models_list:
        if model.name == model_name:
            return model
    return None
