from sklearn import linear_model as Lasso


class LassoRegression(Lasso):

    def fit(self, X, y):
        super().fit(self, X, y)

    def predict(self, X):
        return super().predict(self, X)
