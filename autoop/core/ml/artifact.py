from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    asset: dict

    def __init__(self, name: str,
            version: float,
            asset_path: str,
            tags: list,
            metadata: dict,
            data,
            type: str):

        super().__init__()
        self.asset["name"] = name
        self.asset["version"] = version



