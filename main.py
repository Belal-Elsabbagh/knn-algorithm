import json
from timeit import default_timer

import pandas as pd
from sklearn.model_selection import train_test_split

from src.distance_functions import *
from src.knn import KNN


def iris_test():
    df = pd.read_csv('data/Iris.csv', index_col='Id')
    k = 3
    label = 'Species'
    return test_model(df, k, label)


def test_model(df, k, label):
    train, test = train_test_split(df, test_size=0.1)
    model = KNN(k, train, label, euclidean)
    res = [1 if df[label][index] == model.classify(row) else 0 for index, row in test.iterrows()]
    return len([i for i in res if i == 1]) / len(res)


def lab_test():
    df = pd.read_csv('data/diabetes.csv')
    no_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    for col in no_zero:
        mean = int(df[col].mean(skipna=True))
        df[col] = df[col].replace(0, mean)
    k = 3
    label = 'Outcome'
    return test_model(df, k, label)



if __name__ == '__main__':
    while True:
        start = default_timer()
        r = iris_test()
        print(f'trained and tested model in {round(default_timer() - start, 2)} seconds with success rate: {r}')
