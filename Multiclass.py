from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]
clf = OneVsRestClassifier(estimator=SVC(random_state=0))
clf.fit(X,y)
p = clf.predict(X)
print(p)

y = LabelBinarizer().fit_transform(y)
print(y)
clf.fit(X, y)
p = clf.predict(X)
print(p)

from sklearn.preprocessing import MultiLabelBinarizer
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)
print(y)
clf.fit(X, y)
p = clf.predict(X)
print(p)
