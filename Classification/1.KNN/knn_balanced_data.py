# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 20:04:12 2020

@author: subham
"""

#loading the libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

#loading the data
dataset=pd.read_csv("bank_1.csv", sep=',')

#getting the missing data
print("The missing values are:",dataset.isnull().sum().sum())

#removing the correlated data
def corr_max(dataset):
  dataset_corr = dataset.corr().abs()
  ones_matrix = np.triu(np.ones(dataset_corr.shape), k=1).astype(np.bool)
  dataset_corr = dataset_corr.where(ones_matrix)
  column_drop = [index for index in dataset_corr.columns if any(dataset_corr[index] > 0.80)]
  dataset=dataset.drop(column_drop, axis=1)
  return dataset

dataset=corr_max(dataset)

#splitting the data into X and Y
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,-1]

#scaling
sc=StandardScaler()
X=sc.fit_transform(X)

#splitting data into training set and test set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3, 
                                               random_state=0)

#fitting the model
knn=KNeighborsClassifier(n_neighbors=24)
knn.fit(X_train,Y_train)

#predicting
pred=knn.predict(X_test)

#calculating metrics
cm=confusion_matrix(pred,Y_test)
clas=classification_report(pred,Y_test)
 
#tunning the k value
accuracy=[]
for index in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=index)
    score=cross_val_score(knn,X,Y,cv=10)
    accuracy.append(score.mean())

#plotting the accuracy and neighbour value
plt.plot(range(1,50),accuracy,color="blue",linestyle='dashed', marker="o", 
         markerfacecolor="red",markersize=10)
plt.title('Accuracy vs. K Value')
plt.xlabel('K_Value')
plt.ylabel('Accuracy')

