from collections import Counter
import numpy as np
import pandas as pd


class KNN:
    dataset: pd.DataFrame = None
    distance_function = None
    k = 1

    def __init__(self, k, dataset, label, distance_function):
        self.k = k
        self.dataset = dataset
        self.label = label
        self.distance_function = distance_function

    def __calculate_distances_from_point(self, _object):
        distances = []
        for group, df in self.dataset.groupby(self.label):
            for index, row in df.iterrows():
                distances.append((self.distance_function(row, _object), group))
        return distances

    def __get_knn_labels(self, _object):
        distances = sorted(self.__calculate_distances_from_point(_object))
        return [i[1] for i in distances[:self.k]]

    def classify(self, _object):
        lst = self.__get_knn_labels(_object)
        return max(set(lst), key=lst.count)

    def regress(self, _object):
        return np.mean(self.__get_knn_labels(_object))
