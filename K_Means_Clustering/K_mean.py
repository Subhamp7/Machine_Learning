# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 22:09:28 2020

@author: subham
"""

#importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
from sklearn.metrics import silhouette_score

#loading the dataset
dataset=pd.read_csv("credit_card.csv")
details=dataset.describe()

#replacing the NaN value with the mean of respective column
dataset["MINIMUM_PAYMENTS"]=dataset["MINIMUM_PAYMENTS"].fillna(dataset["MINIMUM_PAYMENTS"].mean())
dataset["CREDIT_LIMIT"]=dataset["CREDIT_LIMIT"].fillna(dataset["CREDIT_LIMIT"].mean())

#removing the column costomer id
dataset=dataset.iloc[:,1:]

#to deal with outliers we are converting the values in range
list_50000=['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
            'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']

list_15=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY',
         'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
         'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT']

list_400=['CASH_ADVANCE_TRX', 'PURCHASES_TRX']

for index in list_50000:
    dataset['Temp']=0
    dataset.loc[((dataset[index]>0     ) & (dataset[index]<=500   )),'Temp']=1
    dataset.loc[((dataset[index]>500   ) & (dataset[index]<=1000  )),'Temp']=2
    dataset.loc[((dataset[index]>1000  ) & (dataset[index]<=2500  )),'Temp']=3
    dataset.loc[((dataset[index]>2500  ) & (dataset[index]<=5000  )),'Temp']=4
    dataset.loc[((dataset[index]>5000  ) & (dataset[index]<=10000 )),'Temp']=5
    dataset.loc[(dataset[index]>10000  ),'Temp']=6
    dataset[index]=dataset['Temp']
    
for index in list_15:
    dataset['Temp']=0
    dataset.loc[((dataset[index]>0)  &(dataset[index]<=0.1)),'Temp']=1
    dataset.loc[((dataset[index]>0.1)&(dataset[index]<=0.2)),'Temp']=2
    dataset.loc[((dataset[index]>0.2)&(dataset[index]<=0.3)),'Temp']=3
    dataset.loc[((dataset[index]>0.3)&(dataset[index]<=0.4)),'Temp']=4
    dataset.loc[((dataset[index]>0.4)&(dataset[index]<=0.5)),'Temp']=5
    dataset.loc[((dataset[index]>0.5)&(dataset[index]<=0.6)),'Temp']=6
    dataset.loc[((dataset[index]>0.6)&(dataset[index]<=0.7)),'Temp']=7
    dataset.loc[((dataset[index]>0.7)&(dataset[index]<=0.8)),'Temp']=8
    dataset.loc[((dataset[index]>0.8)&(dataset[index]<=0.9)),'Temp']=9
    dataset.loc[((dataset[index]>0.9)&(dataset[index]<=1.0)),'Temp']=10
    dataset[index]=dataset['Temp']
    
for index in list_400:
    dataset['Temp']=0
    dataset.loc[((dataset[index]>0)&(dataset[index]<=5)),'Temp']=1
    dataset.loc[((dataset[index]>5)&(dataset[index]<=10)),'Temp']=2
    dataset.loc[((dataset[index]>10)&(dataset[index]<=15)),'Temp']=3
    dataset.loc[((dataset[index]>15)&(dataset[index]<=20)),'Temp']=4
    dataset.loc[((dataset[index]>20)&(dataset[index]<=30)),'Temp']=5
    dataset.loc[((dataset[index]>30)&(dataset[index]<=50)),'Temp']=6
    dataset.loc[((dataset[index]>50)&(dataset[index]<=100)),'Temp']=7
    dataset.loc[((dataset[index]>100)),'Temp']=8
    dataset[index]=dataset['Temp']
    
#droping the temp column used for iteration
dataset=dataset.drop('Temp', axis=1)

#scaling down the dataset
sc=StandardScaler()
dataset=sc.fit_transform(dataset)

#applying K mean using 3 clusters
kmn=KMeans(n_clusters=2)
kmn.fit(dataset)
pred=kmn.predict(dataset)
sil_score=silhouette_score(dataset,pred)

#calculating the scores, distortion and inertia
score=[]
distortions=[]
inertia=[]
k=range(2,20)

for index in k: 
    kmn=KMeans(n_clusters=index)
    kmn.fit(dataset)
    pred=kmn.predict(dataset)
    sil_score=silhouette_score(dataset,pred)
    score.append(sil_score)
    inertia.append(kmn.inertia_)
    distortions.append(sum(np.min(cdist(dataset, kmn.cluster_centers_, 'euclidean'), axis=1)) / dataset.shape[0])
    
fig = plt.figure(2,2,1)
ax1 = fig.add_subplot()
ax1.plot(inertia,'bx-')
ax1.set_xlabel("K value")
ax1.set_ylabel("Inertia")
ax1.set_title("The Elbow method showing the optimal K value using Inertia value")

ax2 = fig.add_subplot(2,2,2)
ax2.plot(k,distortions,'bx-')
ax2.set_xlabel("K value")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("The Elbow method showing the optimal K value using Silhouette score")

plt.show() 