from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
print(data.shape) #(150, 4) (n_samples, n_features)
# print(iris.DESCR)

digits = datasets.load_digits()
shape = digits.images.shape
print(shape)
import matplotlib.pyplot as plt
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)

data = digits.images.reshape((digits.images.shape[0], -1))
print(data)

# estimator.fit(data)
# estimator = Estimator(param1=1, param2=2)
# estimator.param1
# estimator.estimated_param_
# When data is fitted with an estimator, parameters are estimated from
# the data at hand. All the estimated parameters are attributes of the
#  estimator object ending by an underscore:
