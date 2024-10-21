
from abc import abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal


class Model():
    type: str

    def __init__(self):
        pass

    def __str__(self):
        return f"Model(type={self.type})"

    @abstractmethod
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        pass  # your code (attribute and methods) here

    @abstractmethod
    def predict(self, test_X: np.ndarray) -> np.ndarray:
        pass

    def to_artifact(self, name: str) -> Artifact:
        return Artifact(name=name, model=deepcopy(self))
