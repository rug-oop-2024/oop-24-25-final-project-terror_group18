
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
    '''_type: Literal["classification", "regression"] = (
        Field(default_factory=Literal["classification", "regression"]))'''

    def __init__(self):
        self._models.append(self)

    def __str__(self):
        return f"Model(type={self._type})"
    
    # def __iter__(self):
    #     # return iter(self._models)
    
    # def __next__(self):
        # return next(self._models)
    
    def __iter__(self):
        self._index = 0  # Reset index when starting a new iteration
        return self  # Return the instance as an iterator

    def __next__(self):
        if self._index < len(self._models):
            result = self._models[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration  # End of iteration

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
