# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:49:42 2020

@author: subham
"""
#importing required libararies
import os
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#checking if .data file is available, if yes it will be converted to .csv
if(path.exists("wine.data")):
    my_file = 'wine.data'
    base = os.path.splitext(my_file)[0]
    os.rename(my_file, base + '.csv')
    
#loading the dataset( Source UCI)
dataset=pd.read_csv("wine.csv", header=None)

#splitting the dataset into dependent and independent sets
X=dataset.iloc[:,1:].values
Y=dataset.iloc[:,0].values

#splitting the X and Y to training set and test set
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.3, random_state=0)

#fitting the regression model
LR=LinearRegression()
LR.fit(X_train, Y_train)

#predicting the value of Y test
Y_pred=LR.predict(X_test).astype(int)

#combining the value of Y_test(the actual data) and Y_pred(the predicted data)
Final_comparision = pd.DataFrame({'Actual_Data': Y_test, 'Predicted_Data': Y_pred})

#visualizing the actual and predicted data
Final_comparision.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#printing the required details
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

#finding the value of correct prediction
comparison_column = np.where(Final_comparision['Actual_Data'] == Final_comparision['Predicted_Data'], True, False)
print('The model predicted',np.count_nonzero(comparison_column), 'correct values out of' ,comparison_column.size ,'.')
print('Completed')