from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def lasso_model_train(X, y):
    return

clf = linear_model.Lasso(alpha=0.1)
x = [[0,0], [1, 1], [2, 2]]
y = [0, 1, 2]
clf.fit(x, y)

print(clf)
print(clf.coef_)

mse = mean_squared_error(y, clf.predict(x))
print(mse)