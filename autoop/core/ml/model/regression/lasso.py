from sklearn.linear_model import Lasso
import numpy as np
from autoop.core.ml.model.base_model import Model



class LassoRegression(Lasso, Model):

    def __init__(self, alpha=0.1):
        Lasso().__init__(alpha)
        Model().__init__()
        self.type = "regression"
        self.name = "Lasso Regression"

    def fit(self, train_X: np.ndarray, train_y: np.ndarray, **kwargs) -> None:
        Lasso().fit(self, train_X, train_y, kwargs)

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        return Lasso().predict(test_X)  # Lasso instance needs to become class var?
