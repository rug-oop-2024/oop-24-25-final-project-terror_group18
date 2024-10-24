from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable
import numpy as np
from overrides import override

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


class Metric(ABC):
    """
    Base class for all metrics.
    """
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number
    _metrics = []

    def __init__(self, name: str):
        self._name = name
        self._metrics.append(self)

    def __call__(self, y_true, y_pred):
        return self._implementation(y_true, y_pred)

    @property
    def name(self):
        return deepcopy(self._name)

    @property
    def metrics(self):
        return deepcopy(self.metrics)

    @abstractmethod
    def _implementation(self, y_true, y_pred):
        pass


# add here concrete implementations of the Metric class

class MeanSquaredError(Metric):

    def __init__(self, name: str):
        super().__init__(name)

    @override
    def _implementation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mse = (y_true - y_pred) ** 2
        return mse.mean()


class ConfusionMatrix(Metric):  # binary classification metric
    def __init__(self, name: str):
        super().__init__(name)

    def find_TP(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        # counts the number of true positives (y = 1, y_pred = 1)
        return np.sum((y_true == 1) & (y_pred == 1))

    def find_FN(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        # counts the number of false negatives (y = 1, y_pred = 0) Type-II error
        return np.sum((y_true == 1) & (y_pred == 0))

    def find_FP(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        # counts the number of false positives (y = 0, y_pred = 1) Type-I error
        return np.sum((y_true == 0) & (y_pred == 1))

    def find_TN(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        # counts the number of true negatives (y = 0, y_pred = 0)
        return np.sum((y_true == 0) & (y_pred == 0))

    def _implementation(self, y_true: np.ndarray, y_pred: np.ndarray) -> list:
        return [[self.find_TP(y_true, y_pred), self.find_FP(y_true, y_pred)],
                [self.find_TN(y_true, y_pred), self.find_FN(y_true, y_pred)]]


class Accuracy(Metric, ConfusionMatrix):

    def __init__(self, name: str):
        super().__init__(name)

    def _implementation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        TP = super().find_TP(y_true, y_pred)
        FP = super().find_FP(y_true, y_pred)
        TN = super().find_TN(y_true, y_pred)
        FN = super().find_FN(y_true, y_pred)

        return TP + TN / TP + TN + FP + FN

class Precision(Metric, ConfusionMatrix):
    """
    Focuses on type I error of False Positives.
    """
    def __init__(self, name: str):
        super().__init__(name)

    def _implementation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        TP = super().find_TP(y_true, y_pred)
        FP = super().find_FP(y_true, y_pred)
        return TP/(TP+FP)


class Recall(Metric, ConfusionMatrix):
    """
    Focuses on type II error of False Negatives.
    """
    def __init__(self, name: str):
        super().__init__(name)

    def _implementation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        TP = super().find_TP(y_true, y_pred)
        FN = super().find_FN(y_true, y_pred)
        return TP/(TP+FN)


class RootMeanSquaredError(Metric, MeanSquaredError):
    def __init__(self, name: str):
        super().__init__(name)

    def _implementation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(super())


MeanSquaredError("mean_squared_error")
Accuracy("accuracy")
ConfusionMatrix("confusion_matrix")
Precision("precision")
Recall("recall")
RootMeanSquaredError("root_mean_squared_error")
