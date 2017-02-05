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
iris_y_test = iris_y[indices[-10:]]

# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
p = knn.predict(iris_X_test)
print(p)
print(iris_y_test)

#linear model
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_)
meanSqrtError = np.mean((regr.predict(diabetes_X_test)-diabetes_y_test**2))
print(meanSqrtError)
score = regr.score(diabetes_X_test, diabetes_y_test)
print(score)
