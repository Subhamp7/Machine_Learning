# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:49:42 2020

@author: subham
"""
#importing required libararies
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


#loading the dataset( Source UCI)
dataset=pd.read_csv("wine.csv", header=None)

#checking for null value
print("There are {} null values".format(dataset.isnull().sum().sum()))

#splitting the dataset into dependent and independent sets
X=dataset.iloc[:,1:].values
Y=dataset.iloc[:,0].values

#scaling the X datasset
sc=StandardScaler()
X=sc.fit_transform(X)

#splitting the X and Y to training set and test set
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.3, random_state=0)

#
DTC=DecisionTreeClassifier()

DTC.fit(X_train,Y_train)

pred=DTC.predict(X_test)

cm=confusion_matrix(pred,Y_test)


print('\nAccuracy: {:.2f}\n'.format(accuracy_score(Y_test, pred)))

print('\nClassification Report\n')
print(classification_report(Y_test, pred, target_names=['Wine Type 1', 'Wine Type 2', 'Wine Type 3']))


plot_tree(DTC, filled=True)
