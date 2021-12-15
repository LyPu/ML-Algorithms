# coding: utf-8
from numpy import ndarray
from math import sqrt
from collections import Counter
import numpy as np


class KNeighbors(object):
    def __init__(self) -> None:
        """
        Do not modify this function.
        """
        self.predict_result = []

    def knn_classify(self, k: int, x_train: ndarray, y_train: ndarray, x_new: ndarray) -> int:
        """
        In this function you need to fit the input data and label to the KNN classifier
        :param k: Input data which format is int
        :param x_train: Input data which format is ndarray
        :param y_train: Input label which format is ndarray
        :param x_new: Input data which format is ndarray
        :return Returns the sorted result of x_new, as int
        """
        # write your code here
        #(1) Calculate the distance between the new data x_new and each sample data in the whole training data
        distances = [sqrt(np.sum((x_new - x) ** 2)) for x in x_train]
        #(2) Sort the distances and return the corresponding indexes
        nearest = np.argsort(distances)
        #(3) Take the nearest k tag values (classification)
        top_k_y = []
        for i in range(k):
            top_k_y.append(y_train[nearest[i]])
        #(4) count the number of samples belonging to each classification
        votes = Counter(top_k_y)
        #(5) Return the results of the classification that belongs to the largest number of samples
        return votes.most_common(1)[0][0]

    def predict_data(self, x_train: ndarray, y_train: ndarray, x_test: ndarray) -> list:
        """
        In this function you need to use KNN classifier to predict input test data
        :param x_train: Input data which format is ndarray
        :param y_train: Input label which format is ndarray
        :param x_test: Input data which format is ndarray
        :return: The predicted label for input test data which format is list
        """
        # write your code here
        for i in range(x_test.shape[0]):
            self.predict_result.append(self.knn_classify(5, x_train, y_train, x_test[i]))
        return self.predict_result