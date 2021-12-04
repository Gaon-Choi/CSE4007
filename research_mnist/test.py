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
from openTSNE.sklearn import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
MAX_ITER = 1000000

mnist = datasets.load_digits()

# flatten the images
n_samples = len(mnist.images)
data = mnist.data.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(data, mnist.target, test_size = 0.3, shuffle=False)

pca = TSNE(n_components=2)
X2D = pca.fit_transform(X_train)
# X2D = pca.transform(X_train)
X2D_ = pca.transform(X_test)

clf = sklearn.neural_network.MLPClassifier(
        solver='sgd', alpha=1e-5, hidden_layer_sizes=(64, 10, 10), max_iter=MAX_ITER, activation='relu'
    )
clf.fit(X2D, y_train)

grid_size = 500
A, B = np.meshgrid(np.linspace(X2D_[:, 0].min(), X2D_[:, 0].max(), grid_size),
                   np.linspace(X2D_[:, 1].min(), X2D_[:, 1].max(), grid_size))
C = clf.predict( np.hstack([A.reshape(-1, 1), B.reshape(-1, 1)]) ).reshape(grid_size, grid_size)
plt.contourf(A, B, C, alpha=0.3, cmap=plt.cm.gnuplot2)
predicted = clf.predict(X2D_)
accuracy = (predicted == y_test).mean()
plt.xlabel("Accuracy: " + str(round(accuracy, 2)))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'yellow', 'orange', 'purple'
for i, c, label in zip(range(len(X2D_)), colors, mnist.target_names):
    plt.scatter(X2D_[y_test == i, 0], X2D_[y_test == i, 1], c=c, label=label, s=10)

plt.title("Decision Boundary : MLP classifier(with ReLU)")
plt.show()