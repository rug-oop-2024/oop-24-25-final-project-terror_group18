"""
List of available models.
"""

from autoop.core.ml.model.classification.logistic_regression import (
    LogisticRegressor)
from autoop.core.ml.model.classification.knn import KNearestNeighbors
from autoop.core.ml.model.classification.support_vector_machines import (
    SupportVectorClassifier)


__all__ = [LogisticRegressor, KNearestNeighbors, SupportVectorClassifier]
