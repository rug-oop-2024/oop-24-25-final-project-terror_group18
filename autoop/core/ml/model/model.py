
from abc import abstractmethod, ABC

from pydantic import Field

from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal


class Model(ABC):   # Artifact?
    name: str = "model"
    type: Literal["classification", "regression"] = (
        Field(default_factory=Literal["classification", "regression"]))

    '''def __init__(self):
        pass'''

    def __str__(self):
        return f"Model(type={self.type})"

    @abstractmethod
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, test_X: np.ndarray) -> np.ndarray:
        pass

    def to_artifact(self, name: str) -> Artifact:
        return Artifact(name=name, model=deepcopy(self))
