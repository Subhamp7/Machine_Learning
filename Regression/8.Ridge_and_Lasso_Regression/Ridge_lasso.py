# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:45:57 2020

@author: subham
"""

#loading the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression ,Ridge, Lasso

#loading the dataset
boston=load_boston()
dataset=pd.DataFrame(load_boston().data,columns=load_boston().feature_names)
dataset["Dependent"]=load_boston().target

#splitting data into dependent and independent data
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,-1]

#finding the mse value after applying LR
lr=LinearRegression()
MSE=cross_val_score(lr,X,Y,scoring="neg_mean_squared_error", cv=5)
print(np.mean(MSE))

#appliying Ridge
ridge=Ridge()
parameters={"alpha":[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 30, 40, 50, 60, 70, 80, 90, 100]}
regressor=GridSearchCV(ridge,parameters,scoring="neg_mean_squared_error", cv=5)
regressor.fit(X,Y)
print(regressor.best_params_)
print(regressor.best_score_)

#appliying Lasso
lasso=Lasso()
parameters={"alpha":[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 30, 40, 50, 60, 70, 80, 90, 100]}
regressor=GridSearchCV(lasso,parameters,scoring="neg_mean_squared_error", cv=5)
regressor.fit(X,Y)
print(regressor.best_params_)
print(regressor.best_score_)