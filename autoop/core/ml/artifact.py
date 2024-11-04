import pandas as pd
from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    #asset: dict = Field(default_factory=dict)
    id: str = Field(default_factory=str)
    data: bytes = Field(default_factory=bytes)
    name: str = Field(default_factory=str)
    version: str = Field(default_factory=str)
    asset_path: str = Field(default_factory=str)
    tags: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    type: str = Field(default_factory=str)

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
        self.type = type

        for key, value in kwargs:
            setattr(self, key, value)

    def read(self) -> bytes:
        """
        Read artifact data
        """
        return self.data

    def save(self, data: bytes):
        """
        Args:
            data: bytes data to save

        Returns: None

        """
        #self.data = base64.b64encode(data)

