from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """
    The base class for all features.
    """
    name: str = Field(default_factory=str)
    type: Literal["categorical", "numerical"] = (
        Field(default_factory=Literal["categorical", "numerical"]))

    def __init__(
            self, name: str = None,
            type: Literal["categorical", "numerical"] = "categorical") -> None:
        """
        The constructor for the Feature class.
        :param name: str
        :param type: Literal["categorical", "numerical"]
        :return: None
        """
        super().__init__(name=name, type=type)

    def __str__(self) -> str:
        """
        The string representation of the feature.
        :return: str
        """
        return (f"name= {self.name},"
                f"type= {self.type}")
