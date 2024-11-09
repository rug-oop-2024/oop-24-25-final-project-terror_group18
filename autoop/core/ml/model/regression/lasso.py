from sklearn.linear_model import Lasso
import numpy as np
from autoop.core.ml.model.base_model import Model



class LassoRegression(Model):

    def __init__(self, alpha=0.1):
        super().__init__()
        self.type = "regression"
        self.name = "Lasso Regression"
        self._model = Lasso(alpha)

    def fit(self, train_X: np.ndarray, train_y: np.ndarray, **kwargs) -> None:
        self._model.fit(train_X, train_y, **kwargs)
        #self._model.coef_

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        return self._model.predict(test_X)

    def shape(self):
        return self._model.shape