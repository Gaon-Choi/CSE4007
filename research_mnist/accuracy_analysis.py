# Standard scientific python imports
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

''' models '''
models = [
    # logistic regression
    sklearn.linear_model.LogisticRegression(max_iter=5000),

    # Fisher Discriminant Analysis (FDA)
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis(),

    # Support Vector Machine (SVM) - "One vs Rest"
    sklearn.svm.LinearSVC(max_iter=5000, dual=False),

    # Support Vector Machine (SVM) - "One vs One"
    sklearn.svm.SVC(kernel = 'linear', max_iter=5000),

    # K-neighbors Classifier
    sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1),

    # MLP classifier
    sklearn.neural_network.MLPClassifier(
        solver='sgd', alpha=1e-5, hidden_layer_sizes=(64, 10, 10), max_iter=5000
    )
]

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

for clf in models:
    print(clf)
    prev = time.time()
    clf.fit(X_train, y_train)
    elap = time.time()-prev
    print("\t", "Elapsed time: ", elap)
    predicted = clf.predict(X_test)
    accuracy = (predicted == y_test).mean()
    print("\t", "Accuracy: ", accuracy)

    elapsed_time.append(elap)
    acc.append(accuracy)

model_name = [
    'Logistic\nRegression',
    'Linear\nDiscriminant\nAnalysis',
    'Linear SVC',
    'SVC',
    'KNeighbors\nClassifier',
    'MLPClassifier',
]
acc = np.array(acc)
acc *= 100
fig, ax = plt.subplots(1)
ax.bar(model_name, acc, color='blue')
for i, v in enumerate(acc):
    ax.text(i-0.18, v+0.1, round(v, 3), color='black')
ax.set_title("Accuracy Analysis of ML models")
ax.set_ylabel("elapsed time (unit: %)")
ax.set_ylim([85, 100])
plt.tight_layout()
plt.show()