from sklearn import linear_model
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pickle

def at_lasso_model_train(X, y):
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X, y)
    return clf

def at_boostree_model_train(X, y, params):
    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X, y)
    return reg

def save_sklearn_pickle_model(model, pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    print('Saving model in {}'.format(pkl_filename))

def load_sklearn_pickle_model(pkl_filename):
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model



# x = [[0, 0], [1, 1], [2, 2], [3, 3]]
# y = [0, 1, 2, 3]
#
# # clf = at_lasso_model_train(x, y)
# # print(clf)
# # print(clf.coef_)
# #
# # mse = mean_squared_error(y, clf.predict(x))
# # print(mse)
#
# params = {'n_estimators': 1000,
#           'max_depth': 2,
#           'min_samples_split': 3,
#           'learning_rate': 0.01,
#           'verbose': True,
#           'loss': 'ls'}
#
# reg = at_boostree_model_train(X=x, y=y, params=params)
# # pickle_file_name = 'tree.pkl'
# # # save_sklearn_pickle_model(reg, pickle_file_name)
# # # load_reg = load_sklearn_pickle_model(pickle_file_name)
# # #
# # # print(load_reg.predict(x))
# #
# # mse = mean_squared_error(y, reg.predict(x))
# # print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

