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
                 type: str = "N/A", **kwargs):
        super().__init__()
        self.id = base64.b64encode(
            asset_path.encode('utf-8')
        ).decode('utf-8')

        self.asset["artifact_id"] = self.id
        self.asset["name"] = name
        self.asset["version"] = version
        self.asset["asset_path"] = asset_path
        self.asset["tags"] = tags
        self.asset["metadata"] = metadata
        self.asset["data"] = data
        self.asset["type"] = type

        for key, value in kwargs:
            setattr(self, key, value)

    def read(self) -> bytes:
        """
        Read artifact data
        """
        return base64.b64decode(self.asset["artifact_id"])

    def save(self, data: bytes):
        """
        Args:
            data:

        Returns:

        """
        self.asset["data"] = base64.b64encode(data)
