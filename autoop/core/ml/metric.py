from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable
import numpy as np
import pandas as pd
from overrides import override

from autoop.core.ml.ml_type import MLType

METRICS_CLASSIFICATION = [
    "Accuracy",
    "Precision",
    "Recall",
    "Confusion Matrix"
]  

METRICS_REGRESSION = [
    "Mean Squared Error",
    "Root Mean Squared Error"
]


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
    _name: str

    def __init__(self) -> None:
        self._metrics.append(self)

    def __call__(self, y_true, y_pred) :
        return self.evaluate(y_true, y_pred)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def metrics(self):
        return deepcopy(self.metrics)

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass


# add here concrete implementations of the Metric class

class MeanSquaredError(Metric):

    def __init__(self):
        super().__init__()
        self.type = "regression"
        self.name = "Mean Squared Error"

    @override
    def evaluate(self, y_true: Any, y_pred: Any) -> float:
        mse = (y_true - y_pred) ** 2
        return mse.mean()


class ConfusionMatrix(Metric):

    def __init__(self):
        super().__init__()
        self.type = "classification"
        self._matrix = None
        self.name = "Confusion Matrix"

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
            sum = 0
            for j in range(self._matrix[i]):
                sum += self._matrix[i][j]
            false_negatives.append(sum - self._matrix[i][i])
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
        return len(y_true)**2 - (self.find_FN(y_true, y_pred) +
                                 self.find_FP(y_true, y_pred) +
                                 self.find_TP(y_true, y_pred))

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        keys = list(dict.fromkeys(y_pred))
        n = len(keys)
        matrix = np.zeros((n, n))
        for pdata, tdata in y_pred, y_true:
            matrix[keys.index(pdata)][keys.index(tdata)] += 1
        return matrix


class Accuracy(Metric):

    def __init__(self):
        super().__init__()
        self.type = 'classification'
        self.name = "Accuracy"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return (np.sum(y_true == y_pred)/len(y_true))/100

class Precision(ConfusionMatrix):
    """
    Focuses on type I error of False Positives.
    """
    def __init__(self):
        super().__init__()
        self.name = "Precision"

    @override
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        TP = super().find_TP(y_true, y_pred)
        FP = super().find_FP(y_true, y_pred)
        return TP/(TP+FP)


class Recall(ConfusionMatrix):
    """
    Focuses on type II error of False Negatives.
    """
    def __init__(self):
        super().__init__()
        self.name = "Recall"

    @override
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        TP = super().find_TP(y_true, y_pred)
        FN = super().find_FN(y_true, y_pred)
        return TP/(TP+FN)


class RootMeanSquaredError(MeanSquaredError):
    def __init__(self):
        super().__init__()
        self.name = "Root Mean Squared Error"

    @override
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(super())


MeanSquaredError()
Accuracy()
ConfusionMatrix()
Precision()
Recall()
RootMeanSquaredError()
