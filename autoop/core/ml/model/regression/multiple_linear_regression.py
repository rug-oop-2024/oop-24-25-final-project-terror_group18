from sklearn.linear_model import LinearRegression
import numpy as np
from autoop.core.ml.model.base_model import Model



class MultipleLinearRegression(LinearRegression, Model):

    def __init__(self):
        super().__init__()
        self.type = "regression"
        self.name = "Multiple Linear Regression"
        self._lr = LinearRegression()

    def fit(self, train_X: np.ndarray, train_y: np.ndarray, **kwargs) -> None:
        self._lr.fit(train_X, train_y, kwargs)

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        return self._lr.predict(test_X)
