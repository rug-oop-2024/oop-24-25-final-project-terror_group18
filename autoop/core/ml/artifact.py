from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    asset: dict

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

    # data= artifacts["scalar"] ... in pipeline.py??
    # data= artifacts["encoder"] ... in pipeline.py??


