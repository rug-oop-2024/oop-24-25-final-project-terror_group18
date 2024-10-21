import numpy as np
from sklearn.linear_model import LogisticRegression

from autoop.core.ml.model import Model


class BinaryLogisticRegression(LogisticRegression, Model):

    def __init__(self):
        super().__init__()
        LogisticRegression().__init__()
        Model().__init__()
        self.type = "classification"
        self.name = "Logistic Regressor"

    def fit(self, train_x: np.ndarray, train_y: np.array,
            sample_weight=None) -> None:
        super().fit(self, train_x, train_y, sample_weight)

    def predict(self, test_x: np.ndarray) -> float:
        return super().predict(test_x)
