from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X,y)

# dump use pickle
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])
print(y[0])


# with jobslib
from sklearn.externals import joblib
joblib.dump(clf,'file.pk1')
clf3 = joblib.load('file.pk1')
clf3.predict(X[0:1])
print(y[0])
