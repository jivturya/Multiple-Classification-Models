# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:35:06 2022

@author: patzo
"""

import scipy.io as spio
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

path=os.getcwd()

#import true labels
#labels=spio.loadmat(r'{}'.format(path+'\Pathare_Chinmay_HW3\label.mat'),squeeze_me=True)['trueLabel']
xtrain=pd.DataFrame(spio.loadmat(r'{}'.format(path+'\Pathare_Chinmay_HW4\mnist_10digits.mat'),squeeze_me=True)['xtrain']/255)
ytrain=pd.DataFrame(spio.loadmat(r'{}'.format(path+'\Pathare_Chinmay_HW4\mnist_10digits.mat'),squeeze_me=True)['ytrain'])
xtest=pd.DataFrame(spio.loadmat(r'{}'.format(path+'\Pathare_Chinmay_HW4\mnist_10digits.mat'),squeeze_me=True)['xtest']/255)
ytest=pd.DataFrame(spio.loadmat(r'{}'.format(path+'\Pathare_Chinmay_HW4\mnist_10digits.mat'),squeeze_me=True)['ytest'])


#KNN
random_seed = 42
xtrain_knn=xtrain.sample(5000, random_state=random_seed)
ytrain_knn=ytrain.sample(5000, random_state=random_seed)

 
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(xtrain_knn, ytrain_knn)
ypred_knn=neigh.predict(xtest)

print(metrics.classification_report(ytest, ypred_knn))
print(metrics.confusion_matrix(ytest, ypred_knn))


#Logistic Regression
clf = LogisticRegression(random_state=0).fit(xtrain, ytrain)
ypred_lr=clf.predict(xtest)

print(metrics.classification_report(ytest, ypred_lr))
print(metrics.confusion_matrix(ytest, ypred_lr))


#SVM
random_seed = 42
xtrain_knn=xtrain.sample(5000, random_state=random_seed)
ytrain_knn=ytrain.sample(5000, random_state=random_seed)

svm=SVC(C=1.0,gamma='auto').fit(xtrain_knn,ytrain_knn)
ypred_svm=svm.predict(xtest)

print(metrics.classification_report(ytest, ypred_svm))
print(metrics.confusion_matrix(ytest, ypred_svm))



#SVM Kernel
random_seed = 42
xtrain_knn=xtrain.sample(5000, random_state=random_seed)
ytrain_knn=ytrain.sample(5000, random_state=random_seed)

svm_ker=SVC(kernel='poly').fit(xtrain_knn,ytrain_knn)
ypred_svm_ker=svm.predict(xtest)

print(metrics.classification_report(ytest, ypred_svm_ker))
print(metrics.confusion_matrix(ytest, ypred_svm_ker))


#Neural Network
nn=MLPClassifier(random_state=1,hidden_layer_sizes=(20,10)).fit(xtrain,ytrain)

ypred_nn=nn.predict(xtest)

print(metrics.classification_report(ytest, ypred_nn))
print(metrics.confusion_matrix(ytest, ypred_nn))