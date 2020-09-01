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

list_400=['CASH_ADVANCE_TRdataset', 'PURCHASES_TRdataset']

for indedataset in list_50000:
    dataset['Temp']=0
    dataset.loc[((dataset[indedataset]>0     ) & (dataset[indedataset]<=500   )),'Temp']=1
    dataset.loc[((dataset[indedataset]>500   ) & (dataset[indedataset]<=1000  )),'Temp']=2
    dataset.loc[((dataset[indedataset]>1000  ) & (dataset[indedataset]<=2500  )),'Temp']=3
    dataset.loc[((dataset[indedataset]>2500  ) & (dataset[indedataset]<=5000  )),'Temp']=4
    dataset.loc[((dataset[indedataset]>5000  ) & (dataset[indedataset]<=10000 )),'Temp']=5
    dataset.loc[(dataset[indedataset]>10000  ),'Temp']=6
    dataset[indedataset]=dataset['Temp']
    
for indedataset in list_15:
    dataset['Temp']=0
    dataset.loc[((dataset[indedataset]>0)  &(dataset[indedataset]<=0.1)),'Temp']=1
    dataset.loc[((dataset[indedataset]>0.1)&(dataset[indedataset]<=0.2)),'Temp']=2
    dataset.loc[((dataset[indedataset]>0.2)&(dataset[indedataset]<=0.3)),'Temp']=3
    dataset.loc[((dataset[indedataset]>0.3)&(dataset[indedataset]<=0.4)),'Temp']=4
    dataset.loc[((dataset[indedataset]>0.4)&(dataset[indedataset]<=0.5)),'Temp']=5
    dataset.loc[((dataset[indedataset]>0.5)&(dataset[indedataset]<=0.6)),'Temp']=6
    dataset.loc[((dataset[indedataset]>0.6)&(dataset[indedataset]<=0.7)),'Temp']=7
    dataset.loc[((dataset[indedataset]>0.7)&(dataset[indedataset]<=0.8)),'Temp']=8
    dataset.loc[((dataset[indedataset]>0.8)&(dataset[indedataset]<=0.9)),'Temp']=9
    dataset.loc[((dataset[indedataset]>0.9)&(dataset[indedataset]<=1.0)),'Temp']=10
    dataset[indedataset]=dataset['Temp']
    
for indedataset in list_400:
    dataset['Temp']=0
    dataset.loc[((dataset[indedataset]>0)&(dataset[indedataset]<=5)),'Temp']=1
    dataset.loc[((dataset[indedataset]>5)&(dataset[indedataset]<=10)),'Temp']=2
    dataset.loc[((dataset[indedataset]>10)&(dataset[indedataset]<=15)),'Temp']=3
    dataset.loc[((dataset[indedataset]>15)&(dataset[indedataset]<=20)),'Temp']=4
    dataset.loc[((dataset[indedataset]>20)&(dataset[indedataset]<=30)),'Temp']=5
    dataset.loc[((dataset[indedataset]>30)&(dataset[indedataset]<=50)),'Temp']=6
    dataset.loc[((dataset[indedataset]>50)&(dataset[indedataset]<=100)),'Temp']=7
    dataset.loc[((dataset[indedataset]>100)),'Temp']=8
    dataset[indedataset]=dataset['Temp']
    
#droping the temp column used for iteration
dataset=dataset.drop('Temp', adatasetis=1)

#scaling down the dataset
sc=StandardScaler()
dataset=sc.fit_transform(dataset)

#applying K mean using 3 clusters
kmn=KMeans(n_clusters=12)
kmn.fit(dataset)
pred=kmn.predict(dataset)
sil_score=silhouette_score(dataset,pred)

#calculating the scores, distortion and inertia
score=[]
distortions=[]
inertia=[]
k=range(2,40)

for indedataset in k: 
    kmn=KMeans(n_clusters=indedataset)
    kmn.fit(dataset)
    pred=kmn.predict(dataset)
    sil_score=silhouette_score(dataset,pred)
    score.append(sil_score)
    inertia.append(kmn.inertia_)
    distortions.append(sum(np.min(cdist(dataset, kmn.cluster_centers_, 'euclidean'), adatasetis=1)) / dataset.shape[0])
    
#visulizing the inertia and k value
plt.plot(inertia,'bdataset-')
plt.datasetlabel("K value")
plt.ylabel("Inertia")
plt.title("The Elbow method showing the optimal K value using Inertia value")
plt.show() 

#visulizing the Silhouette Score and k value
plt.plot(k,distortions,'bx-')
plt.xlabel("K value")
plt.ylabel("Silhouette Score")
plt.title("The Elbow method showing the optimal K value using Silhouette score")
plt.show() 
