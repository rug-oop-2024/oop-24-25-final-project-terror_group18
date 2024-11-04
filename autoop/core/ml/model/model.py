
from abc import abstractmethod, ABC

from pydantic import Field

from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

from autoop.core.ml.ml_type import MLType


class Model(ABC, MLType):

    _models = []
    _name: str = "model"
    '''_type: Literal["classification", "regression"] = (
        Field(default_factory=Literal["classification", "regression"]))'''

    def __init__(self):
        self._models.append(self)

    def __str__(self):
        return f"Model(type={self.type})"

    @property
    def models(self):
        return self._models
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @abstractmethod
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, test_X: np.ndarray) -> np.ndarray:
        pass

    def to_artifact(self, name: str) -> Artifact:
        return Artifact(name=name, model=deepcopy(self))
