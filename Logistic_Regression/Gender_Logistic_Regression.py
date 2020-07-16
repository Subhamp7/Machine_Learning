# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 21:42:00 2020

@author: subham
"""
#loading the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn import metrics

#loading the dataset
dataset=pd.read_csv("gender_classification.csv")

#checking for Nan Values:
print(dataset.isnull().sum())

#splitting dataset into independet and dependent data
X=dataset.iloc[:,:4]
Y=dataset.iloc[:,-1]

#printing the unique values
for index in X:
    print("\nFor {} the unique values are :\n".format(index),X[index].unique())
    
#encoding using LabelEncoder
encoder_Y=LabelEncoder()
Y=encoder_Y.fit_transform(Y)

#encoding the data and droping the first value
X=pd.get_dummies(X,drop_first=True)

#splitting data into test and train data
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

#applying LR
lr=LogisticRegression()
lr.fit(x_train,y_train)

#predicting using the lr regressor
pred=lr.predict(x_test)

#confusion matrix
cm=metrics.confusion_matrix(y_test,pred)
print("\nThe confusion matrix is:\n",cm)

#printing the accuracy,precision and recall
print("\nAccuracy:",metrics.accuracy_score(y_test, pred))
print("\nPrecision:",metrics.precision_score(y_test, pred))
print("\nRecall:",metrics.recall_score(y_test, pred))

#AUC curve
y_pred_proba = lr.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

#test inputs
x_t=pd.DataFrame()
x_t=pd.get_dummies(x_t,drop_first=True)
pred_t=lr.predict(x_t)
