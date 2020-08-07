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


#loading the dataset( Source UCI)
dataset=pd.read_csv("wine.csv", header=None)

#checking for null value
print("There are {} null values".format(dataset.isnull().sum().sum()))

#splitting the dataset into dependent and independent sets
X=dataset.iloc[:,1:].values
Y=dataset.iloc[:,0].values

#splitting the X and Y to training set and test set
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.3, random_state=0)

