import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print(np.unique(iris_y))

# KNN (K Nearest Neighbors) classification
#split iris data in train and test data

np.random.seed(0)
indices = np.random.permutation(len(iris_X))
print(indices[-10:])
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
