from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNearestNeighbors(KNeighborsClassifier):

    def __init__(self, n_neighbors=5):
        super().__init__(n_neighbors=n_neighbors)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        super().fit(self, train_x, train_y)

    def predict(self, test_x: np.ndarray) -> float:
        return super().predict(test_x)
