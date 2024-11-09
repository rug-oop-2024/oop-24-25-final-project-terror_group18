from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
from typing import Any
import numpy as np
from copy import deepcopy
from typing import Literal, List
import uuid
from autoop.core.ml.ml_type import MLType


class Model(ABC, MLType):
    _model = None
    _name: str = "model"

    def __init__(self):
        self.id = str(uuid.uuid4())

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


    @abstractmethod
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        return None

    @abstractmethod
    def predict(self, test_X: np.ndarray) -> Any:
        return None

    def to_artifact(self, name: str) -> Artifact:
        try:
            return Artifact(name=name,
                            asset_path=name,
                            model=self._model,
                            coeffs=self._model.coef_)
        except AttributeError:
            return Artifact(name=name,
                            asset_path=name,
                            model=self._model)
