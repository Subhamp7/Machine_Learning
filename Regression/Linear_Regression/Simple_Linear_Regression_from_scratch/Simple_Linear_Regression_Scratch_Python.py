# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:29:24 2020

@author: subham
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from math import sqrt

#importing dataset and renaming as Fire and Theft Salary
dataset=(pd.read_csv('Salary_Data.csv')).rename(columns={'YearsExperience': 'Experience', 'Salary':'Salary'})

"""==========Creating all the Functions=========="""

#splitting the data into X and Y
def split(Data):
    dataset_X=Data.iloc[:,0].to_numpy()
    dataset_Y=Data.iloc[:,1].to_numpy()
    return [dataset_X , dataset_Y]

#to calculate mean of a list of numbers
def mean(values):
    return sum(values)/len(values)

#to calculate the variance of a list of numbers
def variance(values):
    mean_for_variance=mean(values)
    return sum( [ (x-mean_for_variance)**2 for x in values] )

#to calculate the covariance of a list of numbers
def covariance(values_X, values_Y):
    mean_X, mean_Y = mean(values_X) , mean(values_Y)
    covar=0
    for index in range(len(values_X)):
        covar += (values_X[index] - mean_X) * (values_Y[index] - mean_Y)
    return covar
        
# to calculate the coefficients
def coefficients(Dataset):
    values_X, values_Y  = split(Dataset)
    mean_X, mean_Y      = mean(values_X) , mean(values_Y)
    coefficient         = covariance(values_X, values_Y) / variance(values_X)
    intercept           = mean_Y - coefficient * mean_X
    return [intercept , coefficient]

#splitting the data to traning set and test set
def train_test_split(dataset, test_size):
    split_index    = int(len(dataset)*test_size)
    dataset_sample= dataset.sample(frac=1).reset_index(drop=True)
    train          = dataset_sample.iloc[split_index:,:]
    test           = dataset_sample.iloc[:split_index,:]
    return train, test

#building the simple linear regression
def simple_linear_regression(Train, Test, continious):
    prediction=[]
    intercept , coefficient = coefficients(Train)
    values_X, values_Y  = split(Test)
    for index in values_X:
        predicted= intercept + coefficient * index
        if(continious==True):
            prediction.append(int(predicted))
        else:
            prediction.append(round(predicted),2)
    return prediction

#to calculate the RMSE
def rmse(actual, predicted):
	return (np.sqrt(np.mean(((actual - predicted) ** 2))))

"""=========Appyling dataset values to the function created=========="""

#splitting the data into training data and test data
train_set, test_set= train_test_split(dataset, 0.3)

#getting the values for the test set
actual_X_test,  actual_Y_test  = split(test_set)
actual_X_train, actual_Y_train = split(train_set)

#applying regression (True for Continious data)
test_pred   =simple_linear_regression(train_set, test_set, continious=True) 
train_pred  =simple_linear_regression(train_set, train_set, continious=True) 

#calculating the RMSE 
print('The Root Mean Square Error of Train is : %.3f' %rmse(actual_X_train, train_pred))
print('The Root Mean Square Error of Test is : %.3f'  %rmse(actual_X_test, test_pred))


#plotting the real value and predicted value for training data
plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.8)
actual     =plt.scatter(actual_X_train, actual_Y_train, zorder=1)
predicted  =plt.scatter(actual_X_train, train_pred,     zorder=2)
plt.legend([actual, predicted], ['Actual', 'Predicted'])
plt.title ('Salary vs Years of Experinece for training set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#plotting the real value and predicted value for test data
plt.subplot(2,1,2)
actual     =plt.scatter(actual_X_test, actual_Y_test, zorder=1)
predicted  =plt.scatter(actual_X_test, test_pred,     zorder=2)
plt.legend([actual, predicted], ['Actual', 'Predicted'])
plt.title ('Salary vs Years of Experinece for test set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()














