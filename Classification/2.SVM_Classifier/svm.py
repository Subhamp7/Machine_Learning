# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:37:42 2020

@author: subham
"""
#loading the libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


#loading the dataset
data=pd.read_csv("bank_1.csv")

#checking the count of missing data
print("The count of missing data is :",data.isnull().sum().sum())

sns.pairplot(data,hue="TARGET CLASS")

#splitting the dependent and independent variables
X=data.iloc[:,:-1]
Y=data.iloc[:,-1]

#scaling
sc=StandardScaler()
X=sc.fit_transform(X)

#splitting data into test and train data
x_train, x_test, y_train,y_test=train_test_split
(X,Y,test_size=0.25,random_state=0)

#creating object for svm
svm=SVC(kernel='linear')
svm.fit(x_train,y_train)

pred=svm.predict(x_test)

cm=confusion_matrix(pred,y_test)