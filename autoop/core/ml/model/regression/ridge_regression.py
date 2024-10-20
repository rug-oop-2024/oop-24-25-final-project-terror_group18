from sklearn.linear_model import Ridge
#RIDGE MODEL NAME FILE CHANGE!!!!!!

class RidgeRegression(Ridge):

    def fit(self, train_X, train_y):
        super().fit(self, train_X, train_y)

    def predict(self, test_X):
        return super().predict(self, test_X)