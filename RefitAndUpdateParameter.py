import numpy as np
from sklearn.svm import SVC

rng = np.random.RandomState(0)
X = rng.rand(100, 10) #100 X 10 vector
print(X[0:2]) # print 0 - 2 elements

y = rng.binomial(1, 0.5, 100) # generate 100 0/1 digit with 0.5 probility
print(y)

X_test = rng.rand(5,10)
print(X_test)


clf =  SVC()
clf.set_params(kernel='linear').fit(X,y)
p = clf.predict(X_test)
print(p)

clf.set_params(kernel='rbf').fit(X,y)
p2 = clf.predict(X_test)
print(p2)
