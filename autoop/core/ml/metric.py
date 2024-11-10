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

    #@override
    def evaluate(self, y_true: Any, y_pred: Any) -> float:
        """
        The evaluate method for the Mean Squared Error metric class.
        :param y_true: Any
        :param y_pred: Any
        :return: float
        """
        #Y_true is any / not np.ndarray???
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

    def _generate_matrix(self, y_true: np.ndarray, y_pred: np.ndarray)-> np.ndarray:
        """
        Generates confusion matrix
        :param y_true: ground truth data
        :param y_pred: observations
        :return: ndarray representing the matrix
        """

            # Flatten y_true and y_pred if they are multidimensional
        y_true = y_true.ravel() if y_true.ndim > 1 else y_true
        y_pred = y_pred.ravel() if y_pred.ndim > 1 else y_pred

        # keys = list(set(y_pred))  # Use set instead of dict.fromkeys for unique items
        keys = list(set(y_true).union(set(y_pred)))
        n = len(keys)
        matrix = np.zeros((n, n), dtype=int)

        for pdata, tdata in zip(y_pred, y_true):  # Iterate over pairs of predictions and true values
            matrix[keys.index(pdata)][keys.index(tdata)] += 1

        self._matrix = matrix  # Store the matrix in the instance for later use
        return matrix
        # keys = list(dict.fromkeys(y_pred))
        # n = len(keys)
        # matrix = np.zeros((n, n))
        # for pdata, tdata in zip(y_pred, y_true):
        #     matrix[keys.index(pdata)][keys.index(tdata)] += 1
        # return matrix

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
        #for i in range(self._matrix):
            true_positives.append(self._matrix[i][i])
        return np.mean(true_positives) #sum or mean??

    def find_FN(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        # counts the number of false negatives (y = 1, y_pred = 0) Type-II error
        # return np.sum((y_true == 1) & (y_pred == 0))
        self._check_matrix(y_true, y_pred)
        false_negatives = []
        # for i in range(self._matrix):
        for i in range(self._matrix.shape[0]):
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

    #@override
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Evaluates confusion matrix
        :param y_true: ground truth data
        :param y_pred: observations
        :return: ndarray representing the matrix
        """
        self._check_matrix(y_pred, y_true)
        return deepcopy(self._matrix)




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

    # @override CHANGED NAME!!! from evaluate to evaluate_precision...
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

    # @override
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        TP = super().find_TP(y_true, y_pred)
        FN = super().find_FN(y_true, y_pred)
        return TP/(TP+FN)


class RootMeanSquaredError(MeanSquaredError):
    def __init__(self):
        super().__init__()
        self.name = "Root Mean Squared Error"

    # @override
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(super())


MeanSquaredError()
Accuracy()
ConfusionMatrix()
Precision()
Recall()
RootMeanSquaredError()
