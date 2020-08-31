# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:35:07 2020

@author: subham
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

#importing dataset
dataset=pd.read_csv("wine.csv", header=None)

#dropping the columns with high corrleation
def corr_max(dataset,corr_value=0.8):
  dataset_corr = dataset.corr().abs()
  ones_matrix = np.triu(np.ones(dataset_corr.shape), k=1).astype(np.bool)
  dataset_corr = dataset_corr.where(ones_matrix)
  column_drop = [index for index in dataset_corr.columns if any(dataset_corr[index] > corr_value)]
  dataset=dataset.drop(column_drop, axis=1)
  return dataset

dataset=corr_max(dataset)

#splitting the data intodependent and independent sets
Y=dataset.iloc[:,:1].values
X=dataset.iloc[:,1:].values

#splitting the data into test set and training set
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.23)

#creating object for Random Forest
rf=RandomForestClassifier()
rf.fit(X_train, Y_train)

#predicting
y_pred=rf.predict(X_test)

#validation
cm=confusion_matrix(Y_test, y_pred)

#classification report
report=classification_report(Y_test, y_pred)

#accuracy
accuracy=accuracy_score(Y_test, y_pred)

'''
The main parameters used by a Random Forest Classifier are:

criterion = the function used to evaluate the quality of a split.
max_depth = maximum number of levels allowed in each tree.
max_features = maximum number of features considered when splitting a node.
min_samples_leaf = minimum number of samples which can be stored in a tree leaf.
min_samples_split = minimum number of samples necessary in a node to cause node splitting.
n_estimators = number of trees in the ensamble.'''

#hyperperemeter tunning manual

#creating object for Random Forest
rf_1=RandomForestClassifier(criterion="entropy")
rf_1.fit(X_train, Y_train)

#predicting
y_pred_1=rf_1.predict(X_test)

#validation
cm_1=confusion_matrix(Y_test, y_pred_1)

#classification report
report_1=classification_report(Y_test, y_pred_1)

#accuracy
accuracy_1=accuracy_score(Y_test, y_pred_1)

#randmized searchcv
from sklearn.model_selection import RandomizedSearchCV
n_estimators=[int(index) for index in np.linspace(start = 200, stop = 2000, num = 10)]
criterion=["gini","entropy"]
max_depth=[int(x) for x in np.linspace(10, 1000,10)]
min_samples_split=[2, 5, 10,14]
min_samples_leaf=[1, 2, 4,6,8]
max_features= ['auto', 'sqrt','log2']

random_grid={'n_estimators':               n_estimators,
             'criterion':                  criterion,
             'max_depth':                  max_depth,
             'min_samples_split':          min_samples_split,
             'min_samples_leaf':           min_samples_leaf,
             'max_features':               max_features}

rf_2=RandomForestClassifier()
rf_random=RandomizedSearchCV(estimator=rf_2, param_distributions=random_grid,
                             n_iter=100,cv=3,verbose=2,random_state=100,n_jobs=-1)
rf_random.fit(X_train, Y_train)

#getting the best estimators
random_model=rf_random.best_estimator_

#validating the model
y_pred_random=random_model.predict(X_test)

cm_random=confusion_matrix(Y_test, y_pred_random)

report_random=classification_report(Y_test, y_pred_random)

accuracy_random=accuracy_score(Y_test, y_pred_random)


#grid searchcv
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': [rf_random.best_params_['criterion']],
    'max_depth': [rf_random.best_params_['max_depth']],
    'max_features': [rf_random.best_params_['max_features']],
    'min_samples_leaf': [rf_random.best_params_['min_samples_leaf'], 
                         rf_random.best_params_['min_samples_leaf']+2, 
                         rf_random.best_params_['min_samples_leaf'] + 4],
    'min_samples_split': [rf_random.best_params_['min_samples_split'] - 2,
                          rf_random.best_params_['min_samples_split'] - 1,
                          rf_random.best_params_['min_samples_split'], 
                          rf_random.best_params_['min_samples_split'] +1,
                          rf_random.best_params_['min_samples_split'] + 2],
    'n_estimators': [rf_random.best_params_['n_estimators'] - 200, rf_random.best_params_['n_estimators'] - 100, 
                     rf_random.best_params_['n_estimators'], 
                     rf_random.best_params_['n_estimators'] + 100, rf_random.best_params_['n_estimators'] + 200]
}

rf_3=RandomForestClassifier()
rf_grid=GridSearchCV(estimator=rf_3, param_grid=param_grid,
                             cv=3, verbose=2, n_jobs=-1)
rf_grid.fit(X_train, Y_train)

grid_model=rf_grid.best_estimator_
#validating the model
y_pred_grid=grid_model.predict(X_test)

cm_grid=confusion_matrix(Y_test, y_pred_grid)

report_grid=classification_report(Y_test, y_pred_grid)

accuracy_grid=accuracy_score(Y_test, y_pred_grid)
