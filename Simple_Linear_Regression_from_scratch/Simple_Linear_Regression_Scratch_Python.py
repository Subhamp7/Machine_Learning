# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:29:24 2020

@author: subham
"""
#import libraries
from random import randrange
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset and renaming as Fire and Theft Salary
dataset=(pd.read_csv('Salary_Data.csv')).rename(columns={'YearsExperience': 'Experience', 'Salary':'Salary'})

#splitting the data into X and Y
def split(Data):
    dataset_X=Data.iloc[:,0]
    dataset_Y=Data.iloc[:,1]
    return [dataset_X , dataset_Y]

#to calculate mean of a list of numbers
def mean(values):
    return round(sum(values)/len(values),2)

#to calculate the variance of a list of numbers
def variance(values):
    mean_for_variance=mean(values)
    return round( sum( [ (x-mean_for_variance)**2 for x in values] ), 2)

#to calculate the covariance of a list of numbers
def covariance(values_X, values_Y):
    mean_X, mean_Y = mean(values_X) , mean(values_Y)
    covar=0
    for index in range(len(values_X)):
        covar += (values_X[index] - mean_X) * (values_Y[index] - mean_Y)
    return round(covar, 2)
        
# to calculate the coefficients
def coefficients(Dataset):
    values_X, values_Y = split(Dataset)
    mean_X, mean_Y  = mean(values_X) , mean(values_Y)
    coefficient     = covariance(values_X, values_Y) / variance(values_X)
    intercept       = mean_Y - coefficient * mean_X
    return [intercept , coefficient]

#building the simple linear regression
def simple_linear_regression(Train, Test):
    predictions=[]
    intercept , coefficient = coefficients(Train)
    for index in Test:
        pred= intercept + coefficient * index[0]
        predictions.append(pred)
    return predictions
    
#splitting the data to traning set and test set

print(('The Covariance of Experience and Salary is {}').format(coefficients(dataset)))










