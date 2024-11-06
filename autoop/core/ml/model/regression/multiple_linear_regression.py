from sklearn.linear_model import LinearRegression
import numpy as np
from autoop.core.ml.model.base_model import Model



class MultipleLinearRegression(LinearRegression, Model):

    def __init__(self):
        LinearRegression().__init__()
        Model().__init__()
        self.type = "regression"
        self.name = "Multiple Linear Regression"

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        super().fit(self, train_X, train_y)

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        return super().predict(self, test_X)
