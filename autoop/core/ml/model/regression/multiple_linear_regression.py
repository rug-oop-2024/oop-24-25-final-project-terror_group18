from sklearn.linear_model import LinearRegression
import numpy as np


class MultipleLinearRegression(LinearRegression):

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        super().fit(self, train_X, train_y)

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        return super().predict(self, test_X)
