from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    asset: dict
    id: str

    def __init__(self, name: str,  # default values fix
                 version: str = "N/A",
                 asset_path: str = "N/A",
                 tags: list = [],
                 metadata: dict = {},
                 data: str = "N/A",
                 type: str = "N/A"):

        super().__init__()
        self.asset["name"] = name
        self.asset["version"] = version
        self.asset["asset_path"] = asset_path
        self.asset["tags"] = tags
        self.asset["metadata"] = metadata
        self.asset["data"] = data
        self.asset["type"] = type
        id = 


    def read(self) -> bytes:
        """
        Read artifact data
        """
        return base64.b64decode(self.asset["artifact_id"])
    
    def save(self, data: bytes):
        self.asset["data"] = base64.b64encode(data)
