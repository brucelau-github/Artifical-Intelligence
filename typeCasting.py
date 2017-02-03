import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
X = rng.rand(10,2000)
X = np.array(X, dtype='float32')
print(X.dtype)

# by default variable will cast to float64
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X.dtype)


from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
print(iris)
clf = SVC()
clf.fit(iris.data, iris.target)
print(list(clf.predict(iris.data[:3])))
clf.fit(iris.data, iris.target_names[iris.target])
print(list(clf.predict(iris.data[:3])))
