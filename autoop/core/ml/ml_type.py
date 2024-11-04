from typing import Literal

from pydantic import Field


class MLType:
    _type: Literal["classification", "regression"] = (
        Field(default_factory=Literal["classification", "regression"]))

    @property
    def type(self):
        return self._type
