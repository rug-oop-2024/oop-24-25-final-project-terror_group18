from sklearn.svm import SVC
import numpy as np


class Support_Vector_Classifier(SVC):

    def __init__(self, kernel='linear'):
        super().__init__(kernel=kernel)

    def fit(self, train_x: np.ndarray, train_y: np.array, sample_weight=None):
        super().fit(self, train_x, train_y)

    def predict(self, test_x):
        super().predict(test_x)
