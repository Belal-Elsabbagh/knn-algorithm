from collections import Counter
import numpy as np


class KNN:
    dataset: dict[int:list] = None
    distance_function = None
    k = 1

    def __init__(self, k, dataset, distance_function):
        self.k = k
        self.dataset = dataset
        self.distance_function = distance_function

    def __calculate_distances_from_point(self, _object):
        distances = []
        for group, features in self.dataset.items():
            distances += [(self.distance_function(point, _object), group) for point in features]
        return distances

    def __get_knn_labels(self, _object):
        distances = sorted(self.__calculate_distances_from_point(_object))
        return [i[1] for i in distances[:self.k]]

    def classify(self, _object):
        lst = self.__get_knn_labels(_object)
        return max(set(lst), key=lst.count)

    def regress(self, _object):
        return np.mean(self.__get_knn_labels(_object))
