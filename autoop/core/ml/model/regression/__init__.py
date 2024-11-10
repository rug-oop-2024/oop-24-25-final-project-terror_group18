"""
List of available regression models.
"""

from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression)
from autoop.core.ml.model.regression.lasso import LassoRegression
from autoop.core.ml.model.regression.ridge_regression import RidgeRegression


__all__ = [MultipleLinearRegression, LassoRegression, RidgeRegression]
