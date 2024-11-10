from sklearn.svm import SVC
import numpy as np
from autoop.core.ml.model.base_model import Model


class SupportVectorClassifier(Model):
    """
    The class for Support Vector Classifier model.
    """
    def __init__(self, kernel: str = 'linear') -> None:
        """
        The constructor for the SupportVectorClassifier class.
        :param kernel: kernel
        :return: None
        """
        super().__init__()
        self.type = "classification"
        self.name = "Support Vector Classifier"
        self._model = SVC(kernel=kernel)

    def fit(self, train_x: np.ndarray, train_y: np.array) -> None:
        """
        A method that fits the training data to the model.
        :param train_x: np.ndarray training observations data
        :param train_y: np.ndarray training ground truth labels
        :return: None
        """
        self._model.fit(train_x, train_y)

    def predict(self, test_x: np.ndarray) -> float:
        """
        A method that predicts the labels for the test data.
        :param test_x: np.ndarray test data
        :return: predicted labels float
        """
        return self._model.predict(test_x)
