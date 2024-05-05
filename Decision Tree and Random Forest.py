# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 11:00:49 2022

@author: patzo
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2, random_state=10)

#Build a CART model
cart=DecisionTreeClassifier()
cart_mod=cart.fit(xtrain,ytrain)

#predict using CART model
ypred_cart=cart_mod.predict(xtest)

#Print classification tree
tree.plot_tree(cart_mod)


#Build Random Forest
forest=RandomForestClassifier()
forest_mod=forest.fit(xtrain,ytrain)

#predict using random forest on test data
ypred_forest=forest_mod.predict(xtest)

#Calculate accuracy for both methods
forest_acc=accuracy_score(ypred_forest,ytest)
cart_acc=accuracy_score(ypred_cart,ytest)
print("CART Model Error=",(1-cart_acc)*100)
print("Random Forest Model Error=",(1-forest_acc)*100)


#Creating new model to plot number of trees Vs error rate for both models
acc_forest=[]
acc_cart=[]

for n in range(1,75):
    #build models and fit
    cm = DecisionTreeClassifier(max_depth=n)
    fm = RandomForestClassifier(max_depth=n)
    cm_mod=cm.fit(xtrain,ytrain)
    fm_mod=fm.fit(xtrain,ytrain)
    
    #Predict xtest with each mod
    cm_pred=cm_mod.predict(xtest) 
    fm_pred=fm_mod.predict(xtest)
    
    #calculate acc and add to list
    cm_acc=accuracy_score(cm_pred,ytest)
    fm_acc=accuracy_score(fm_pred,ytest)
    acc_cart.append((1-cm_acc)*100)
    acc_forest.append((1-fm_acc)*100)
    
#plot figure for accuracy om both models
fig = plt.figure()
ax1 = fig.add_subplot(111)

plt.title('Error(%) vs. Tree Size')
plt.xlabel('Tree Size')
plt.ylabel('Error(%)')

ax1.scatter(range(1,75), acc_cart, s=10, c='b', marker="s", label='Decistion Tree')
ax1.scatter(range(1,75),acc_forest, s=10, c='r', marker="o", label='Random Forest')
plt.legend(loc='upper right');
plt.show()

    
    