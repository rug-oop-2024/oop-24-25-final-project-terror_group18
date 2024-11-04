from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable
import numpy as np
import pandas as pd
from overrides import override

from autoop.core.ml.ml_type import MLType

METRICS = [
    "mean_squared_error",
    "root_mean_squared_error"
    "accuracy",
    "precision",
    "recall",
    "confusion_matrix"

]  # add the names (in strings) of the metrics you implement


# https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide

def get_metric(name: str):
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    for metric in Metric.metrics:
        if metric.name == name:
            return metric
    return None


class Metric(ABC, MLType):
    """
    Base class for all metrics.
    """
    # remember: metrics take ground truth and prediction as input and return a real number
    _metrics: list = []

    def __init__(self, name: str):
        self._name = name
        self._metrics.append(self)

    def __call__(self, y_true, y_pred):
        return self.evaluate(y_true, y_pred)

    @property
    def name(self):
        return deepcopy(self._name)

    @property
    def metrics(self):
        return deepcopy(self.metrics)

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass


# add here concrete implementations of the Metric class

class MeanSquaredError(Metric):

    def __init__(self, name: str):
        super().__init__(name)
        self._type = "regression"

    @override
    def evaluate(self, y_true: Any, y_pred: Any) -> float:
        mse = (y_true - y_pred) ** 2
        return mse.mean()


class ConfusionMatrix(Metric):

    def __init__(self, name: str):
        super().__init__(name)
        self._type = "classification"
        self._matrix = None

    def _check_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> :
        if self._matrix is None:
            self.evaluate(y_true, y_pred)

    def find_TP(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        # counts the number of true positives (y = y_pred)
        self._check_matrix(y_true, y_pred)
        true_positives = []
        for i in range(self._matrix):
            true_positives.append(self._matrix[i][i])
        return np.mean(true_positives)

    def find_FN(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        # counts the number of false negatives (y = 1, y_pred = 0) Type-II error
        # return np.sum((y_true == 1) & (y_pred == 0))
        self._check_matrix(y_true, y_pred)
        false_negatives = []
        for i in range(self._matrix):
            for j in range(self._matrix[i])
                false_negatives.append(self._matrix[i][j])
        return np.mean(false_negatives)

    def find_FP(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        # counts the number of false positives (y = 0, y_pred = 1) Type-I error
        self._check_matrix(y_true, y_pred)
        false_positives = []
        for i in range(self._matrix):
            false_positives.append(np.sum(self._matrix[i]) - self._matrix[i][i])
        return np.mean(false_positives)

    def find_TN(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        # counts the number of true negatives (y = 0, y_pred = 0)
        # return np.sum((y_true == 0) & (y_pred == 1))
        self._check_matrix(y_true, y_pred)
        return np.sum((y_true == 0) & (y_pred == 0))

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        keys = list(dict.fromkeys(y_pred))
        n = len(keys)
        matrix = np.zeros((n, n))
        for pdata, tdata in y_pred, y_true:
            matrix[keys.index(pdata)][keys.index(tdata)] += 1
        return matrix


class Accuracy(Metric):

    def __init__(self, name: str):
        super().__init__(name)
        self._type = 'classification'

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return (np.sum(y_true == y_pred)/len(y_true))/100

class Precision(ConfusionMatrix):
    """
    Focuses on type I error of False Positives.
    """
    def __init__(self, name: str):
        super().__init__(name)

    @override
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        TP = super().find_TP(y_true, y_pred)
        FP = super().find_FP(y_true, y_pred)
        return TP/(TP+FP)


class Recall(ConfusionMatrix):
    """
    Focuses on type II error of False Negatives.
    """
    def __init__(self, name: str):
        super().__init__(name)

    @override
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        TP = super().find_TP(y_true, y_pred)
        FN = super().find_FN(y_true, y_pred)
        return TP/(TP+FN)


class RootMeanSquaredError(MeanSquaredError):
    def __init__(self, name: str):
        super().__init__(name)

    @override
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(super())


MeanSquaredError("mean_squared_error")
Accuracy("accuracy")
ConfusionMatrix("confusion_matrix")
Precision("precision")
Recall("recall")
RootMeanSquaredError("root_mean_squared_error")
