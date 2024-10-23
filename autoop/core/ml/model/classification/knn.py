from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from autoop.core.ml.model import Model


class KNearestNeighbors(KNeighborsClassifier, Model):

    def __init__(self, n_neighbors=5):
        KNeighborsClassifier().__init__(n_neighbors=n_neighbors)
        Model().__init__()
        self.type = "classification"
        self.name = "KNN"


    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        super().fit(self, train_x, train_y)

    def predict(self, test_x: np.ndarray) -> float:
        return super().predict(test_x)
