# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 07:44:07 2022

@author: patzo
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import os

path=os.getcwd()
data=pd.read_csv(r'{}'.format(path+'\Pathare_Chinmay_HW4\marriage.csv'),header=None)
#data=pd.read_csv(r'M:\OMSA\ISYE6740\HW4\HW4\data\marriage.csv',header=None)
print(data.head())
x=pd.DataFrame(data.loc[:,0:54])
y=pd.DataFrame(data.loc[:,54:55])

print(x)



#Split into training and testing data
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.80,random_state=30)


#Naive bayes
gnb=GaussianNB()
ypred_nb=gnb.fit(xtrain,ytrain).predict(xtest)

print(metrics.classification_report(ytest, ypred_nb))


#Logistic Regression 
clf = LogisticRegression(random_state=0).fit(xtrain, ytrain)
ypred_lr=clf.predict(xtest)

print(metrics.classification_report(ytest, ypred_lr))

#KNN
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(xtrain, ytrain)
ypred_knn=neigh.predict(xtest)

print(metrics.classification_report(ytest, ypred_knn))


