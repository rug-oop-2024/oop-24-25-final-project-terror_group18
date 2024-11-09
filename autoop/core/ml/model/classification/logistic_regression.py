import numpy as np
from sklearn.linear_model import LogisticRegression

from autoop.core.ml.model.base_model import Model


class LogisticRegressor(Model):

    def __init__(self):
        super().__init__()
        self.type = "classification"
        self.name = "Logistic Regressor"
        self._model = LogisticRegression()

    def fit(self, train_x: np.ndarray, train_y: np.array,
            sample_weight=None) -> None:
        self._model.fit(train_x, train_y, sample_weight)

    def predict(self, test_x: np.ndarray) -> float:
        return self._model.predict(test_x)
