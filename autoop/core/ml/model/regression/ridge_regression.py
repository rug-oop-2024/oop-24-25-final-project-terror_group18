from sklearn.linear_model import Ridge
import numpy as np
from autoop.core.ml.model.base_model import Model


class RidgeRegression(Model):

    def __init__(self):
        super().__init__()
        self.type = "regression"
        self.name = "Ridge Regression"
        self._model = Ridge()

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        self._model.fit(train_X, train_y)
        self._parameters = self._model.get_params()


    def predict(self, test_X: np.ndarray) -> np.ndarray:
        return self._model.predict(test_X)
