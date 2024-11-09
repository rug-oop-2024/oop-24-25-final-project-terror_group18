from abc import abstractmethod, ABC

from pydantic import Field

from autoop.core.ml.artifact import Artifact
from typing import Any
import numpy as np
from copy import deepcopy
from typing import Literal, List

from autoop.core.ml.ml_type import MLType


class Model(ABC, MLType):
    _model = None
    _name: str = "model"
    _parameters = None

    def __str__(self) -> str:
        return f"Model(type={self._type})"

    @property
    def model(self) -> List:
        return deepcopy(self._model)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def parameters(self) -> Any:
        return deepcopy(self._parameters)

    @abstractmethod
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        return None

    @abstractmethod
    def predict(self, test_X: np.ndarray) -> Any:
        return None

    def to_artifact(self, name: str) -> Artifact:
        return Artifact(name=self.name,
                        asset_path=self._type + '_' + self.name,
                        type=self.type,
                        parameters=self._parameters)

