from sklearn.linear_model import LinearRegression


class MultipleLinearRegression(LinearRegression):

    def fit(self, X, y):
        super().fit(self, X, y)

    def predict(self, X):
        return super().predict(self, X)
