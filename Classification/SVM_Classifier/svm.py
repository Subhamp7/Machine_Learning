# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:37:42 2020

@author: subham
"""
#loading the libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report


#loading the dataset
data=pd.read_csv("glass_data.csv")

#checking the count of missing data
print("The count of missing data is :",data.isnull().sum().sum())

#the target output
print("The outputs are:\n",data["Type"].unique())

#plotting corelations
#sns.heatmap(data.corr(),annot=True)

#splitting the dependent and independent variables
X=data.iloc[:,:9]
Y=data.iloc[:,-1]

#splitting data into test and train data
x_train, x_test, y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#creating object for svm
svm=SVC(kernel='linear')
svm.fit(x_train,y_train)

pred=svm.predict(x_test)