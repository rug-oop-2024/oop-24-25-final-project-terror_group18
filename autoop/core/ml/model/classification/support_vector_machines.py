from sklearn.svm import SVC
import numpy as np

from autoop.core.ml.model.base_model import Model


class SupportVectorClassifier(SVC, Model):

    def __init__(self, kernel='linear'):
        SVC().__init__(kernel=kernel)
        Model().__init__()
        self.type = "classification"
        self.name = "Support Vector Classifier"

    def fit(self, train_x: np.ndarray, train_y: np.array, sample_weight=None) -> None:
        super().fit(self, train_x, train_y)

    def predict(self, test_x: np.ndarray) -> float:
        return super().predict(test_x)
