
from abc import abstractmethod, ABC

from pydantic import Field

from autoop.core.ml.artifact import Artifact
from typing import Any
import numpy as np
from copy import deepcopy
from typing import Literal, List

from autoop.core.ml.ml_type import MLType


class Model(ABC, MLType):

    _models = []
    _name: str = "model"

    def __str__(self):
        return f"Model(type={self._type})"

    @property
    def models(self) -> List:
        return deepcopy(self._models)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @abstractmethod
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        return None

    @abstractmethod
    def predict(self, test_X: np.ndarray) -> Any:
        return None

    def to_artifact(self, name: str) -> Artifact:
        return Artifact(name=name, model=deepcopy(self))
