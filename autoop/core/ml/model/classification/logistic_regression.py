import numpy as np
from sklearn.linear_model import LogisticRegression
from autoop.core.ml.model.base_model import Model


class LogisticRegressor(Model):
    """
    The class for Logistic Regression model.
    """
    def __init__(self) -> None:
        """
        The constructor for the LogisticRegressor class.
        :return: None
        """
        super().__init__()
        self.type = "classification"
        self.name = "Logistic Regressor"
        self._model = LogisticRegression()

    def fit(self, train_x: np.ndarray, train_y: np.array,
            sample_weight: np.ndarray = None) -> None:
        """
        A method that fits the training data to the model.
        :param train_x: np.ndarray training observations data
        :param train_y: np.ndarray training ground truth labels
        :return: None
        """
        self._model.fit(train_x, train_y, sample_weight)

    def predict(self, test_x: np.ndarray) -> float:
        """
        A method that predicts the labels for the test data.
        :param test_x: np.ndarray test data
        :return: predicted labels float
        """
        return self._model.predict(test_x)
