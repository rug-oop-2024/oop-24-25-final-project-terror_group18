from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from autoop.core.ml.model.base_model import Model


class KNearestNeighbors(Model):
    """
    The class for K-Nearest Neighbors model.
    """
    def __init__(self, n_neighbors: int = 5) -> None:
        """
        The constructor for the KNearestNeighbors class.
        :param n_neighbors: int number of neighbors
        :return: None
        """
        super().__init__()
        self.type = "classification"
        self.name = "KNN"
        self._model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        A method that fits the training data to the model.
        :param train_x: np.ndarray training data
        :param train_y: np.ndarray training labels
        """
        self._model.fit(train_x, train_y)

    def predict(self, test_x: np.ndarray) -> float:
        """
        A method that predicts the labels for the test data.
        :param test_x: np.ndarray test data
        :return: predicted labels float
        """
        return self._model.predict(test_x)
