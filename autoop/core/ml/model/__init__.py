from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression import LassoRegression
from autoop.core.ml.model.regression import RidgeRegression
from autoop.core.ml.model.classification import LogisticRegression
from autoop.core.ml.model.classification import KNN
from autoop.core.ml.model.classification import SupportVectorClassifier

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


def get_model(model_name: str) -> Model | None:
    """Factory function to get a model by name."""
    # -models[] remains empty always ->fix



    if model_name == "Multiple Linear Regression":
        return MultipleLinearRegression()
    elif model_name == "Lasso Regression":
        return LassoRegression()
    elif model_name == "Ridge Regression":
        return RidgeRegression()
    elif model_name == "Logistic Regressor":
        return LogisticRegression()
    elif model_name == "KNN":
        return KNN()
    elif model_name == "Support Vector Classifier":
        return SupportVectorClassifier()
    

    # try:
    #     for model in models_list:
    #         if model.name == model_name:
    #             return model
    # except Exception as e:
    #     print(type(models_list), models_list)
    #     print(e)
    # return None
