import numpy as np
from sklearn.linear_model import LogisticRegression


class BinaryLogisticRegression(LogisticRegression):

    def __init__(self):
        super().__init__()

    def fit(self, train_x: np.ndarray, train_y: np.array, sample_weight=None):
        super().fit(self, train_x, train_y, sample_weight)

    def predict(self, test_x):
        super().predict(test_x)
