from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any
import numpy as np
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


def get_metric(metric_name: str) -> 'Metric':
    """
    The method that returns a metric instance based on its name.
    :param metric_name: str
    :return: Metric
    """
    if metric_name == "Mean Squared Error":
        return MeanSquaredError()
    elif metric_name == "Root Mean Squared Error":
        return RootMeanSquaredError()
    elif metric_name == "Accuracy":
        return Accuracy()
    elif metric_name == "Precision":
        return Precision()
    elif metric_name == "Recall":
        return Recall()
    elif metric_name == "Confusion Matrix":
        return ConfusionMatrix()


class Metric(ABC, MLType):
    """
    Base class for all metrics.
    """
    _metrics: list = []
    _name: str

    def __init__(self) -> None:
        """
        The constructor for the Metric class.
        :return: None
        """
        self._metrics.append(self)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Any:
        """
        The call method for the Metric class.
        :param y_true: np.ndarray
        :param y_pred: np.ndarray
        :return: Any
        """
        return self.evaluate(y_true, y_pred)

    @property
    def name(self) -> str:
        """
        The getter method for the name of the metric.
        :return: str
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """
        The setter method for the name of the metric.
        :param name: str
        :return: None
        """
        self._name = name

    @property
    def metrics(self) -> list:
        """
        The getter method for the metrics.
        :return: list
        """
        return deepcopy(self._metrics)

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Any:
        """
        The abstract method for the evaluate method.
        :param y_true: np.ndarray
        :param y_pred: np.ndarray
        :return: Any"""
        pass


class MeanSquaredError(Metric):
    """
    The Mean Squared Error metric class.
    """
    def __init__(self) -> None:
        """
        The constructor for the Mean Squared Error metric class.
        :return: None
        """
        super().__init__()
        self.type = "regression"
        self.name = "Mean Squared Error"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        The evaluate method for the Mean Squared Error metric class.
        :param y_true: np.ndarray
        :param y_pred: np.ndarray
        :return: float
        """
        mse = (y_true - y_pred) ** 2
        return mse.mean()


class ConfusionMatrix(Metric):
    """
    The Confusion Matrix metric class.
    """
    def __init__(self) -> None:
        """
        The constructor for the Confusion Matrix metric class.
        :return: None
        """
        super().__init__()
        self.type = "classification"
        self._matrix = None
        self.name = "Confusion Matrix"

    def _generate_matrix(self, y_true: np.ndarray,
                         y_pred: np.ndarray) -> np.ndarray:
        """
        Generates confusion matrix
        :param y_true: np.ndarray
        :param y_pred: np.ndarray
        :return: ndarray representing the matrix
        """

        y_true = y_true.ravel() if y_true.ndim > 1 else y_true
        y_pred = y_pred.ravel() if y_pred.ndim > 1 else y_pred
        keys = list(set(y_true).union(set(y_pred)))
        n = len(keys)
        matrix = np.zeros((n, n), dtype=int)

        for pdata, tdata in zip(y_pred, y_true):
            matrix[keys.index(pdata)][keys.index(tdata)] += 1

        self._matrix = matrix
        return matrix

    def _check_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        # This method checks if the matrix is already created.
        # :param y_true: np.ndarray
        # :param y_pred: np.ndarray
        # :return: None
        """
        if self._matrix is None:
            self._generate_matrix(y_true, y_pred)

    def find_TP(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        """
        This method counts the number of true positives.
        :param y_true: np.ndarray
        :param y_pred: np.ndarray
        :return: int
        """
        self._check_matrix(y_true, y_pred)
        true_positives = []

        for i in range(self._matrix.shape[0]):
            true_positives.append(self._matrix[i][i])
        return np.mean(true_positives)

    def find_FN(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        """
        This method counts the number of false negatives.
        :param y_true: np.ndarray
        :param y_pred: np.ndarray
        :return: int
        """
        self._check_matrix(y_true, y_pred)
        false_negatives = []
        for i in range(self._matrix.shape[0]):
            row_sum = np.sum(self._matrix[i, :])
            fn_count = row_sum - self._matrix[i, i]
            false_negatives.append(fn_count)
        return int(np.sum(false_negatives))

    def find_FP(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        """
        This method counts the number of false positives.
        :param y_true: np.ndarray
        :param y_pred: np.ndarray
        :return: int
        """
        self._check_matrix(y_true, y_pred)
        false_positives = []

        for i in range(self._matrix.shape[0]):
            col_sum = np.sum(self._matrix[:, i])
            fp_count = col_sum - self._matrix[i, i]
            false_positives.append(fp_count)
        return np.sum(false_positives)

    def find_TN(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        """
        This method counts the number of true negatives.
        :param y_true: np.ndarray
        :param y_pred: np.ndarray
        :return: int
        """
        return len(y_true)**2 - (self.find_FN(y_true, y_pred) +
                                 self.find_FP(y_true, y_pred) +
                                 self.find_TP(y_true, y_pred))

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Evaluates confusion matrix
        :param y_true: np.ndarray
        :param y_pred: np.ndarray
        :return: ndarray representing the matrix
        """
        self._check_matrix(y_pred, y_true)
        return deepcopy(self._matrix)


class Accuracy(Metric):
    """
    The Accuracy metric class.
    """
    def __init__(self) -> None:
        """
        The constructor for the Accuracy metric class.
        :return: None
        """
        super().__init__()
        self.type = 'classification'
        self.name = "Accuracy"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluates accuracy
        :param y_true: np.ndarray
        :param y_pred: np.ndarray
        :return: float
        """
        return (np.sum(y_true == y_pred)/len(y_true))/100


class Precision(ConfusionMatrix):
    """
    Focuses on type I error of False Positives.
    """
    def __init__(self) -> None:
        """
        The constructor for the Precision metric class.
        :return: None
        """
        super().__init__()
        self.name = "Precision"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluates precision
        :param y_true: np.ndarray
        :param y_pred: np.ndarray
        :return: float
        """
        TP = super().find_TP(y_true, y_pred)
        FP = super().find_FP(y_true, y_pred)
        return TP/(TP+FP)


class Recall(ConfusionMatrix):
    """
    Focuses on type II error of False Negatives.
    """
    def __init__(self) -> None:
        """
        The constructor for the Recall metric class.
        :return: None
        """
        super().__init__()
        self.name = "Recall"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluates recall
        :param y_true: np.ndarray
        :param y_pred: np.ndarray
        :return: float
        """
        TP = super().find_TP(y_true, y_pred)
        FN = super().find_FN(y_true, y_pred)
        return TP/(TP+FN)


class RootMeanSquaredError(MeanSquaredError):
    """
    The Root Mean Squared Error metric class.
    """
    def __init__(self) -> None:
        """
        The constructor for the Root Mean Squared Error metric class.
        :return: None
        """
        super().__init__()
        self.name = "Root Mean Squared Error"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluates root mean squared error
        :param y_true: np.ndarray
        :param y_pred: np.ndarray
        :return: float
        """
        return np.sqrt(super().evaluate(y_true, y_pred))


MeanSquaredError()
Accuracy()
ConfusionMatrix()
Precision()
Recall()
RootMeanSquaredError()
