from sklearn.linear_model import Ridge
import numpy as np
from autoop.core.ml.model.base_model import Model


class RidgeRegression(Model):
    """
    The Ridge Regression model class.
    """
    def __init__(self) -> None:
        """
        The constructor for the RidgeRegression class.
        :return: None
        """
        super().__init__()
        self.type = "regression"
        self.name = "Ridge Regression"
        self._model = Ridge()

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        """
        A method that fits the training data to the model.
        :param train_X: np.ndarray training observations data
        :param train_y: np.ndarray training ground truth data
        :return: None
        """
        self._model.fit(train_X, train_y)
        self._parameters = self._model.get_params()

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """
        A method that predicts the y-values for the test data.
        :param test_X: np.ndarray test data
        :return: np.ndarray
        """
        return self._model.predict(test_X)
