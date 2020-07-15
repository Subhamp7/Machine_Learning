# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:48:34 2020

@author: subham
"""
# Linear Regression

# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn import metrics

# Importing DataSet
dataset=pd.read_csv("USA_Housing.csv")
X=dataset.iloc[:,:5]
Y=dataset.iloc[:,-2]

#visualizing the data
corr=dataset.corr()
sns.heatmap(corr)

# Splitting data into training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3,random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import  LinearRegression
LR=LinearRegression()
LR.fit(X_train, Y_train)

#printing the slope and intercept
print("Intercept:" ,LR.intercept_)
print("Slope:",LR.coef_)

#passing the X_test value to predict 
pred=LR.predict(X_test)

# Evaluating Model
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, pred)))
print("Accuracy:",metrics.r2_score(Y_test, pred))


X = sm.add_constant(X)
est = sm.OLS(Y, X)
est2 = est.fit()
print("summary()\n",est2.summary())
print("------------------------------------------------------------------\n")
print("pvalues\n",est2.pvalues)
print("------------------------------------------------------------------\n")
print("tvalues\n",est2.tvalues)
print("------------------------------------------------------------------\n")
print("rsquared\n",est2.rsquared)
print("------------------------------------------------------------------\n")
print("rsquared_adj\n",est2.rsquared_adj)

