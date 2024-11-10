from autoop.core.ml.artifact import Artifact
import pandas as pd


class Dataset(Artifact):
    """
    The dataset artifact class.
    """
    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        """
        The constructor for the Dataset class.
        :param args: tuple
        :param kwargs: dict
        :return: None
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str,
                       version: str = "1_0_0") -> 'Dataset':
        """
        The method that creates a dataset artifact from a dataframe.
        :param data: pd.DataFrame
        :param name: str
        :param asset_path: str
        :param version: str
        :return: Dataset
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version
        )

    def read(self) -> pd.DataFrame:
        """
        Read artifact data as a dataframe.
        :return: pd.DataFrame
        """
        return super().read()

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Save artifact data as bytes.
        :param data: pd.DataFrame
        :return: bytes
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
