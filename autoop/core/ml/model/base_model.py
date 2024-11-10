from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
from typing import Any
import numpy as np
from copy import deepcopy
from typing import Literal, List
import uuid
from autoop.core.ml.ml_type import MLType


class Model(ABC, MLType):
    """
    The base class for all models.
    """
    _model = None
    _name: str = "model"

    def __str__(self) -> str:
        """
        The string representation of the model.
        :return: str
        """
        return f"Model(type={self._type})"

    @property
    def model(self) -> List:
        """
        The getter method for the model object.
        :return: List
        """
        return deepcopy(self._model)

    @property
    def name(self) -> str:
        """
        The getter method for the name of the model.
        :return: str
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """
        The setter method for the name of the model.
        :param name: str
        :return: None
        """
        self._name = name

    @abstractmethod
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        """
        The abstract method that fits the training data to the model.
        :param train_X: np.ndarray training observations data
        :param train_y: np.ndarray training ground truth data
        :return: None
        """
        return None

    @abstractmethod
    def predict(self, test_X: np.ndarray) -> Any:
        """
        The abstract method that predicts the y-values for the test data.
        :param test_X: np.ndarray test observations data
        :return: Any
        """
        return None

    def to_artifact(self, name: str) -> Artifact:
        """
        The method that converts the model to an artifact.
        :param name: str
        :return: Artifact
        """
        return Artifact(name=self.name,
                        asset_path=name,
                        type=self._type)
