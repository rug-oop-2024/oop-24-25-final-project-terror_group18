import pandas as pd
from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    #asset: dict = Field(default_factory=dict)
    id: str = Field(default_factory=str)
    data: bytes = Field(default_factory=bytes)
    name: str = Field(default_factory=str)

    def __init__(self, name: str,
                 version: str = "N/A",
                 asset_path: str = "N/A",
                 tags: list = [],
                 metadata: dict = {},
                 data: bytes = "N/A",
                 type: str = "N/A", **kwargs):
        super().__init__()
        encoded_id = base64.b64encode(
            asset_path.encode('utf-8')
        ).decode('utf-8')

        self.id = f"{encoded_id}:{version}"
        self.data = data
        self.name = name
        self.version = version
        self.asset_path = asset_path
        self.tags = tags
        self.metadata = metadata
        self.data = data
        self.type = type

        for key, value in kwargs:
            setattr(self, key, value)

    def read(self) -> bytes:
        """
        Read artifact data
        """
        return self.asset["data"]

    def save(self, data: bytes):
        """
        Args:
            data:

        Returns:

        """
        self.asset["data"] = base64.b64encode(data)
