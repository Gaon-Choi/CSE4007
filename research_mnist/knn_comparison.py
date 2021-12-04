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
from sklearn.manifold import TSNE

# mnist = sklearn.datasets.load_digits()
mnist = datasets.load_digits()
print(mnist.data.shape)
print(mnist['images'].shape)

# flatten the images
n_samples = len(mnist.images)
data = mnist.images.reshape((n_samples, -1))


X_train, X_test, y_train, y_test = train_test_split(data, mnist.target, test_size = 0.3, shuffle=False)

# results
elapsed_time = list()
acc = list()
N = range(1, 11)
for n in range(1, 11):
    # print("KNN with n={n}".format(n=n))
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
    # prev = time.time()
    clf.fit(X_train, y_train)
    # elap = time.time()-prev
    # print("\t", "Elapsed time: ", elap)
    predicted = clf.predict(X_test)
    accuracy = (predicted == y_test).mean()
    # print("\t", "Accuracy: ", accuracy)

    # elapsed_time.append(elap)
    acc.append(accuracy)

plt.plot(N, acc, color='pink')
plt.title("Performance Comparison between KNNs with different N")
plt.xlabel("n"); plt.ylabel("accuracy")
plt.show()