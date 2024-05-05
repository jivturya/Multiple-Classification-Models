# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 12:04:34 2022

@author: patzo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
import os

path=os.getcwd()
data=pd.read_csv(r'{}'.format(path+'\Pathare_Chinmay_HW5\spambase.data'),header=None, na_filter=False)
#data=pd.read_csv(r'M:\OMSA\ISYE6740\HW5\spambase.data',header=None, na_filter=False)
print(data)


x=data[data.columns[:-1]]
y=data[data.columns[57:58]]
print(x.shape,y.shape)


#Count empty values
print("Number of empty cells:", data.isnull().sum().sum())


#Split data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.3, random_state=10)
print("xtrain shape:",xtrain.shape,"ytrain shape",ytrain.shape)


#remove non-spam emails from training data
idx=ytrain.index[ytrain[57]==0].tolist()


ytrain_drop=ytrain.drop(idx)
xtrain_drop=xtrain.drop(idx)
print("xtrain drop shape:",xtrain_drop.shape,"ytrain drop shape:",ytrain_drop.shape)



#Build SVM with RBF Kernel
svm=OneClassSVM(kernel='rbf').fit(xtrain_drop,ytrain_drop)
ypred_svm=svm.predict(xtest)


#measure accuracy for SVM 
svm_acc=accuracy_score(ypred_svm,ytest)
print("Misclassification error rate on one-class SVM (%):",(1-svm_acc)*100)