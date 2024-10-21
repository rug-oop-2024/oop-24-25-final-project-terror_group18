
from typing import List
from pandas.core.dtypes.common import is_numeric_dtype

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: Dataset) -> list[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    df = dataset.read()
    feature_list = []
    for column in df:
        if df.isnull().values.any():
            df.dropna()
            # message?
        feature = Feature(column)
        if is_numeric_dtype(df[column].dtype):
            feature.type = "numeric"
        else:
            feature.type = "categorical"

        feature_list.append(feature)

    return feature_list
