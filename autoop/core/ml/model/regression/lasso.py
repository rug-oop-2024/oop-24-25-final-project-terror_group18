from sklearn.linear_model import Lasso
import numpy as np
from autoop.core.ml.model.base_model import Model



class LassoRegression(Lasso, Model):

    def __init__(self, alpha=0.1):
        super().__init__()
        self.type = "regression"
        self.name = "Lasso Regression"
        self._lasso = Lasso(alpha)

    def fit(self, train_X: np.ndarray, train_y: np.ndarray, **kwargs) -> None:
        self._lasso.fit(self, train_X, train_y, kwargs)

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        return self._lasso.predict(test_X)
