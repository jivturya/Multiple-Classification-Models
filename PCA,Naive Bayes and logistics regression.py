#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
"""
print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.decomposition import PCA
import os

path=os.getcwd()
data=pd.read_csv(r'{}'.format(path+'\Pathare_Chinmay_HW4\marriage.csv'),header=None)
xdata=data[data.columns[:-1]]
y=data[data.columns[54:55]]
print(xdata)


##PCA
pca=PCA(n_components=2)
pca.fit(xdata)
x=pd.DataFrame(pca.transform(xdata))

h = .02  # step size in the mesh

# preprocess dataset, split into training and test part
y=data[data.columns[54:55]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2, random_state=30)

#"""
#Naive Bayes
model=GaussianNB().fit(xtrain,ytrain)
predicted=model.predict(xtest)
print(predicted)

x_min, x_max = x[0].min() - .5, x[0].max() + .5
y_min, y_max = x[1].min() - .5, x[1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(10, 8))
plt.suptitle('Gaussian', fontsize=30)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(xtest[0], xtest[1], c=predicted, edgecolors='k', cmap='viridis')
plt.xlabel('X1', fontsize=20)
plt.ylabel('X2', fontsize=20)



#logistic regression
clf = LogisticRegression(random_state=42).fit(xtrain, ytrain)

predicted_lr=clf.predict(xtest)


x_min, x_max = x[0].min() - .5, x[0].max() + .5
y_min, y_max = x[1].min() - .5, x[1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(10, 8))
plt.suptitle('Logistics', fontsize=30)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(xtest[0], xtest[1], c=predicted_lr, edgecolors='k', cmap='viridis')
plt.xlabel('X1', fontsize=20)
plt.ylabel('X2', fontsize=20)



#KNN
neigh = KNeighborsClassifier(n_neighbors=2).fit(xtrain, ytrain)

predicted_knn=neigh.predict(xtest)


x_min, x_max = x[0].min() - .5, x[0].max() + .5
y_min, y_max = x[1].min() - .5, x[1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = neigh.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(10, 8))
plt.suptitle('KNN', fontsize=30)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(xtest[0], xtest[1], c=predicted_knn, edgecolors='k', cmap='viridis')
plt.xlabel('X1', fontsize=20)
plt.ylabel('X2', fontsize=20)
