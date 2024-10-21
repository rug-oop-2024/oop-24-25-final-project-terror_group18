
from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal


class Model(ABC):
    type: str

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        pass  # your code (attribute and methods) here

    @abstractmethod
    def predict(self, test_X: np.ndarray) -> np.ndarray:
        pass
