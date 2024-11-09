from sklearn.svm import SVC
import numpy as np

from autoop.core.ml.model.base_model import Model


class SupportVectorClassifier(SVC, Model):

    def __init__(self, kernel='linear'):
        super().__init__()
        self.type = "classification"
        self.name = "Support Vector Classifier"
        self._model = SVC(kernel=kernel)

    def fit(self, train_x: np.ndarray, train_y: np.array, sample_weight=None) -> None:
        self._model.fit(self, train_x, train_y)

    def predict(self, test_x: np.ndarray) -> float:
        return self._model.predict(test_x)
