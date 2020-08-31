# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:35:07 2020

@author: subham
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report

#importing dataset
dataset=pd.read_csv("wine.csv", header=None)

#dropping the columns with high corrleation
def corr_max(dataset,corr_value=0.8):
  dataset_corr = dataset.corr().abs()
  ones_matrix = np.triu(np.ones(dataset_corr.shape), k=1).astype(np.bool)
  dataset_corr = dataset_corr.where(ones_matrix)
  column_drop = [index for index in dataset_corr.columns if any(dataset_corr[index] > corr_value)]
  dataset=dataset.drop(column_drop, axis=1)
  return dataset

dataset=corr_max(dataset)

#splitting the data intodependent and independent sets
Y=dataset.iloc[:,:1].values
X=dataset.iloc[:,1:].values

#splitting the data into test set and training set
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.23)

#creating object for Random Forest
rf=RandomForestClassifier()
rf.fit(X_train, Y_train)

#predicting
y_pred=rf.predict(X_test)

#validation
cm=confusion_matrix(Y_test, y_pred)

#classification report
report=classification_report(Y_test, y_pred)

'''
The main parameters used by a Random Forest Classifier are:

criterion = the function used to evaluate the quality of a split.
max_depth = maximum number of levels allowed in each tree.
max_features = maximum number of features considered when splitting a node.
min_samples_leaf = minimum number of samples which can be stored in a tree leaf.
min_samples_split = minimum number of samples necessary in a node to cause node splitting.
n_estimators = number of trees in the ensamble.'''

#hyperperemeter tunning

#creating object for Random Forest
rf_1=RandomForestClassifier(criterion="entropy")
rf_1.fit(X_train, Y_train)

#predicting
y_pred_1=rf_1.predict(X_test)

#validation
cm_1=confusion_matrix(Y_test, y_pred_1)

#classification report
report_1=classification_report(Y_test, y_pred_1)