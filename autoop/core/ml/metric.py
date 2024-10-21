from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable
import numpy as np
from overrides import override

METRICS = [
    "mean_squared_error",
    "accuracy",
    "precision",
    "recall",
    "f1-score",
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

class MSE(Metric):

    def __init__(self, name):
        super().__init__(name)

    @override
    def _implementation(self, y_true, y_pred):
        mse = (y_true - y_pred) ** 2
        return mse.mean()


class Accuracy(Metric):

    def __init__(self, name):
        super().__init__(name)
    def _implementation(self, y_true, y_pred):
        return  # ...


for metric_name in METRICS:
    Metric(metric_name)
