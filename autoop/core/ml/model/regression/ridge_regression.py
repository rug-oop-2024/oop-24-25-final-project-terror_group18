from sklearn.linear_model import Ridge
import numpy as np
from autoop.core.ml.model import Model


class RidgeRegression(Ridge, Model):

    def __init__(self):
        Ridge().__init__()
        Model().__init__()
        self.type = "regression"
        self.name = "Ridge Regression"

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        super().fit(self, train_X, train_y)

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        return super().predict(self, test_X)
