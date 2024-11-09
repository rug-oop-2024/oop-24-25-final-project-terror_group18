import io

import pandas as pd
from pydantic import BaseModel, Field
import base64


class Artifact():
    asset = {}
    id: str
    data: bytes
    name: str
    version: str
    asset_path: str
    tags: list
    metadata: dict
    type: str

    def __init__(self, name: str,
                 version: str = "1",
                 asset_path: str = "NA",
                 tags: list = [],
                 metadata: dict = {},
                 data: bytes = b"NA",
                 type: str = "NA",
                 **kwargs):
        super().__init__()
        encoded_id = base64.b64encode(
            asset_path.encode('utf-8')
        ).decode('utf-8')

        self.id = f"{encoded_id}_{version}"
        #raise ValueError(f"id is {self.id}")
        '''if not isinstance(data, bytes):
            data = base64.b64encode(data)'''
        self.data = data
        self.name = name
        self.version = version
        self.asset_path = asset_path
        self.tags = tags
        self.metadata = metadata
        self.type = type
        self.asset["data"] = data

        #try:
        for key, value in kwargs.items():
            setattr(self, key, value)
        '''except ValueError:
            raise ValueError(f"kwargs are {kwargs}")'''


    def read(self) -> pd.DataFrame:
        """
        Read artifact data
        """
        bytes = self.data
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: bytes):
        """
        Args:
            data: bytes data to save

        Returns: None

        """
        self.data = base64.b64encode(data)

