import matplotlib.pyplot as plt

import sklearn.linear_model
import sklearn.discriminant_analysis
import sklearn.svm
import sklearn.neighbors
import sklearn.neural_network
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.decomposition import PCA

mnist = datasets.load_digits()
print(mnist.data.shape)
print(mnist['images'].shape)

# flatten the images
n_samples = len(mnist.images)
data = mnist.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(data, mnist.target, test_size = 0.3, shuffle=False)

pca = PCA(n_components=2)
X2D = pca.fit_transform(mnist.data)
X2D = pca.transform(mnist.data)
print(type(X2D))
print(X2D.shape)

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'yellow', 'orange', 'purple'
for i, c, label in zip(range(len(mnist.target_names)), colors, mnist.target_names):
    plt.scatter(X2D[mnist.target == i, 0], X2D[mnist.target == i, 1], c=c, label=label, s=10)
plt.legend()
plt.title("PCA Results : MNIST")
plt.show()