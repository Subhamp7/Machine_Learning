# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:29:24 2020

@author: subham
"""
#import libraries
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset and renaming as Fire and Theft Salary
dataset=(pd.read_csv('Salary_Data.csv')).rename(columns={'YearsExperience': 'Experience', 'Salary':'Salary'})

#splitting the data into X and Y
dataset_X=dataset.iloc[:,0]
dataset_Y=dataset.iloc[:,1]

#to calculate mean of a list of numbers
def mean(values):
    return round(sum(values)/len(values),2)

#to calculate the variance of a list of numbers
def variance(values):
    mean_for_variance=mean(values)
    return round( sum( [ (x-mean_for_variance)**2 for x in values] ), 2)

plt.scatter(dataset_X, dataset_Y)





