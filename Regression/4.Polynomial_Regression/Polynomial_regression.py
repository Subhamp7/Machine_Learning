# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 18:04:44 2020

@author: subham
"""
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score

#loading the dataset
dataset=pd.read_csv("Position_Salaries.csv")

#splitting the data to dependent and independent sets
X=dataset.iloc[:,1:2]
Y=dataset.iloc[:,-1]

#applying liner resgression
lr=LinearRegression()
lr.fit(X,Y)

#visulizing the linear fit
plt.scatter(X,Y, color= 'red')
plt.plot(X,lr.predict(X),color= 'blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#applying ploynomial regression
p_reg=PolynomialFeatures(degree=5)#we must provide the degree
lr_1=LinearRegression()
X_poly=p_reg.fit_transform(X)
lr_1.fit(X_poly,Y)

#visulizing the polynomial fit
plt.scatter(X, Y, color = 'red')
plt.plot(X, lr_1.predict(p_reg.fit_transform(X)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# evaluating the model Linear Regression
pred_1=lr.predict(X)
rmse_lr = np.sqrt(mean_squared_error(Y, pred_1 ))
r2_lr = r2_score(Y, pred_1)

# evaluating the model Polynomial Regression
pred_2=lr_1.predict(X_poly)
rmse_pr = np.sqrt(mean_squared_error(Y, pred_2))
r2_pr = r2_score(Y, pred_2)

print("\n")

print("The model performance for Linear Regression")
print("-------------------------------------------")
print("RMSE Linear Regression is {}".format(rmse_lr))
print("R2 score Linear Regression is {}".format(r2_lr))

print("\n")

print("The model performance for Polynomial regression")
print("-------------------------------------------")
print("RMSE Polynomial regression is {}".format(rmse_pr))
print("R2 score Polynomial regression is {}".format(r2_pr))