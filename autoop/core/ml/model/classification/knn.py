from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from autoop.core.ml.model.base_model import Model


class KNearestNeighbors(Model):

    def __init__(self, n_neighbors=5):
        super().__init__()
        self.type = "classification"
        self.name = "KNN"
        self._model = KNeighborsClassifier(n_neighbors=n_neighbors)


    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self._model.fit(train_x, train_y)

    def predict(self, test_x: np.ndarray) -> float:
        return self._model.predict(test_x)
