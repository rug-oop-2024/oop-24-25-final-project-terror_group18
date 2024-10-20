
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    name: str = Field(default_factory=str)
    type: Literal["categorical", "numerical"] = Field(default="categorical")
    # do we need default in Field and in __init__???

    def __init__(self, name: str = None,
                 type: Literal["categorical", "numerical"] = "categorical"):
        super().__init__(name=name, type=type)

    # attributes here

    def __str__(self):
        raise NotImplementedError("To be implemented.")
