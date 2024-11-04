from typing import Literal

from pydantic import Field


class MLType:
    _type: Literal["classification", "regression"] = (
        Field(default_factory=Literal["classification", "regression"]))

    @property
    def type(self) -> Literal["classification", "regression"]:
        return self._type

    @type.setter
    def type(self, value: Literal["classification", "regression"]) -> None:
        self._type = value
