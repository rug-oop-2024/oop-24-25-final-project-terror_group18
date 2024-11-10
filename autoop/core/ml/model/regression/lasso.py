from sklearn.linear_model import Lasso
import numpy as np
from autoop.core.ml.model.base_model import Model


class LassoRegression(Model):
    """
    The Lasso Regression model class."""
    def __init__(self, alpha: float = 0.1) -> None:
        """
        The constructor for the LassoRegression class.
        :param alpha: float
        :return: None
        """
        super().__init__()
        self.type = "regression"
        self.name = "Lasso Regression"
        self._model = Lasso(alpha)

    def fit(self, train_X: np.ndarray, train_y: np.ndarray,
            **kwargs: dict) -> None:
        """
        A method that fits the training data to the model.
        :param train_X: np.ndarray training observations data
        :param train_y: np.ndarray training ground truth data
        :return: None
        """
        self._model.fit(train_X, train_y, **kwargs)

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """
        A method that predicts the y-values for the test data.
        :param test_X: np.ndarray test data
        :return: np.ndarray
        """
        return self._model.predict(test_X)

    def shape(self) -> tuple:
        """
        A method that returns the shape of the training data used
        to fit the model.
        :return: tuple representing the shape of the training data
        """
        return self._model.shape
