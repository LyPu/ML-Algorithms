#!/usr/bin/python
# -*- coding: UTF-8 -*-
# from numpy import ndarray
# import numpy as np
#
# def linear_regression(xtrain: ndarray, ytrain: ndarray, xtest: ndarray) -> float:
#     """
#     :param xtrain: Training set X
#     :param ytrain: Training set Y
#     :param xtest: Testing set X
#     :return: Return the mse you got from model
#     """
#     # -- write your code here --
#     data_points = xtrain.shape[0]
#     xtrain, xtest = normalize_data(xtrain, xtest)
#     xtrain = np.concatenate((np.ones((xtrain.shape[0], 1)), xtrain), axis = 1)
#     xtest = np.concatenate(((np.ones((xtest.shape[0], 1))), xtest), axis = 1)
#
#     theta = fit(xtrain, ytrain, 10000)
#
#     return predict(xtest, theta), theta
#
# def predict(xtest: ndarray, theta: int):
#     return np.matmul(xtest, theta)
#
#
# def fit(xtrain: ndarray, ytrain:ndarray, max_iter: int):
#     data_points = xtrain.shape[0]
#     features = xtrain.shape[1]
#     theta = np.array(np.random.rand(features, 1))
#     eta = 0.1
#
#     for _ in range(max_iter):
#         theta = update_theta(xtrain, theta, ytrain, eta)
#     return theta
#
# def normalize_data(xtrain: ndarray, xtest: ndarray):
#     mean = np.mean(xtrain, axis = 0)
#     std = np.std(xtrain, axis = 0)
#     xtrain = (xtrain - mean) / std
#     xtest = (xtest - mean) / std
#     return xtrain, xtest
#
# def update_theta(xtrain: ndarray, theta: int, ytrain: ndarray, eta: int):
#     data_points = xtrain.shape[0]
#     gradients = 2 / data_points * np.matmul(xtrain.T, np.matmul(xtrain, theta) - ytrain)
#     theta = theta - eta * gradients
#     return theta
#
# def loss_function(xtrain: ndarray, theta: int, ytrain: ndarray):
#     data_points = xtrain.shape[0]
#     temp = (np.matmul(xtrain, theta) - ytrain).reshape(-1)
#     return temp.dot(temp) / data_points


import numpy as np

def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

class LinearRegression:
    def __init__(self, learning_rate = 0.001, n_iters = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.matmul(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.matmul(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_approximated = np.matmul(X, self.weights) + self.bias
        return y_approximated


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


    X, y = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print("MSE: ", mse)

    accuracy = r2_score(y_test, predictions)
    print("Accuracy: ", accuracy)

    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize = (8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s = 10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s = 10)
    plt.plot(X, y_pred_line, color = "black", linewidth = 2, label = "Prediction")
    plt.show()