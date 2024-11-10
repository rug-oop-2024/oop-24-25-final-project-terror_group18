import io
import pandas as pd
import base64


class Artifact():
    """
    The base class for all artifacts.
    """
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
                 **kwargs: dict) -> None:
        """
        The constructor for the Artifact class.

        :param name: str name of the artifact
        :param version: str version of the artifact
        :param asset_path: str path of the artifact
        :param tags: list tags of the artifact
        :param metadata: dict metadata of the artifact
        :param data: bytes data of the artifact
        :param type: str type of the artifact
        :param kwargs: dict kwargs of the artifact
        :return: None
        """
        super().__init__()
        encoded_id = base64.b64encode(
            asset_path.encode('utf-8')
        ).decode('utf-8')

        self.id = f"{encoded_id}_{version}"
        self.data = data
        self.name = name
        self.version = version
        self.asset_path = asset_path
        self.tags = tags
        self.metadata = metadata
        self.type = type
        self.asset["data"] = data

        for key, value in kwargs.items():
            setattr(self, key, value)


    def read(self) -> pd.DataFrame:
        """
        Read artifact data as a dataframe.
        :return: pd.DataFrame
        """
        bytes = self.data
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: bytes) -> None:
        """
        Save artifact data.
        :param data: bytes
        :return: None
        """
        self.data = base64.b64encode(data)
