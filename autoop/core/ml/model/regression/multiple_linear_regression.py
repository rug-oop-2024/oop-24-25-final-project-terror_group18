from sklearn.linear_model import LinearRegression
import numpy as np
from autoop.core.ml.model.base_model import Model


class MultipleLinearRegression(Model):
    """
    The Multiple Linear Regression model class.
    """
    def __init__(self) -> None:
        """
        The constructor for the MultipleLinearRegression class.
        :return: None
        """
        super().__init__()
        self.type = "regression"
        self.name = "Multiple Linear Regression"
        self._model = LinearRegression()

    def fit(self, train_X: np.ndarray, train_y: np.ndarray,
            **kwargs: dict) -> None:
        """
        A method that fits the training data to the model.
        :param train_X: np.ndarray training observations data
        :param train_y: np.ndarray training ground truth data
        :param kwargs: dict
        :return: None
        """
        self._model.fit(train_X, train_y, **kwargs)
        self._parameters = self._model.get_params()

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """
        A method that predicts the y-values for the test data.
        :param test_X: np.ndarray test data
        :return: np.ndarray
        """
        return self._model.predict(test_X)
