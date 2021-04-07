from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def at_lasso_model_train(X, y):
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X, y)
    return clf


x = [[0, 0], [1, 1], [2, 2], [3, 3]]
y = [0, 1, 2, 3]

clf = at_lasso_model_train(x, y)
print(clf)
print(clf.coef_)

mse = mean_squared_error(y, clf.predict(x))
print(mse)