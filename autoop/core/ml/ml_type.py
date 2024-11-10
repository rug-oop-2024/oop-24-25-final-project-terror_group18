from typing import Literal
from pydantic import Field


class MLType:
    """
    The base class for all ML types."""
    _type: Literal["classification", "regression"] = (
        Field(default_factory=Literal["classification", "regression"]))

    @property
    def type(self) -> Literal["classification", "regression"]:
        """
        The getter method for the type of the ML model."""
        return self._type

    @type.setter
    def type(self, value: Literal["classification", "regression"]) -> None:
        """
        The setter method for the type of the ML model."""
        self._type = value
