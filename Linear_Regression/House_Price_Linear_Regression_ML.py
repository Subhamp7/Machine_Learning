# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:48:34 2020

@author: subham
"""
# Linear Regression

# Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt

# Importing DataSet
dataset=pd.read_csv("USA_Housing.csv")
X=dataset.iloc[:,:5]
Y=dataset.iloc[:,-2]

# Splitting data into training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3,random_state=101)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import  LinearRegression
LR=LinearRegression()
LR.fit(X_train, Y_train)

# Evaluating Model
print("Intercept:" ,LR.intercept_)

# Plotting the prediction
Prediction= LR.predict(X_test)
plt.scatter(Y_test, Prediction , color = 'blue')
plt.show()



