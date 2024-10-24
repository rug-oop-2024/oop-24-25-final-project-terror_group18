
from abc import abstractmethod, ABC

from pydantic import Field

from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal


class Model(ABC):   # Artifact?

    _models = []
    name: str = "model"
    _type: Literal["classification", "regression"] = (
        Field(default_factory=Literal["classification", "regression"]))

    def __init__(self):
        self._models.append(self)

    def __str__(self):
        return f"Model(type={self._type})"

    @property
    def models(self):
        return self._models

    @abstractmethod
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, test_X: np.ndarray) -> np.ndarray:
        pass

    def to_artifact(self, name: str) -> Artifact:
        return Artifact(name=name, model=deepcopy(self))
