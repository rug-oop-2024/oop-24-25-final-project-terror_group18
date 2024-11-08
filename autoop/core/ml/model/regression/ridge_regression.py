from sklearn.linear_model import Ridge
import numpy as np
from autoop.core.ml.model.base_model import Model


class RidgeRegression(Model):

    def __init__(self):
        super().__init__()
        self.type = "regression"
        self.name = "Ridge Regression"
        self._ridge = Ridge()

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        self._ridge.fit(train_X, train_y)

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        return self._ridge.predict(test_X)
