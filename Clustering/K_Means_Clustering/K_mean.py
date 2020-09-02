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
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

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
dataset=dataset.drop('Temp', axis=1)

#removing the correlated data
def corr_max(dataset):
  dataset_corr = dataset.corr().abs()
  ones_matrix = np.triu(np.ones(dataset_corr.shape), k=1).astype(np.bool)
  dataset_corr = dataset_corr.where(ones_matrix)
  column_drop = [index for index in dataset_corr.columns if any(dataset_corr[index] > 0.80)]
  dataset=dataset.drop(column_drop, axis=1)
  return dataset

dataset=corr_max(dataset)

#scaling down the dataset
sc=StandardScaler()
dataset=sc.fit_transform(dataset)

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
    
#visulizing the inertia and k value
plt.plot(k,inertia,'bx-')
plt.xlabel("K value")
plt.ylabel("Inertia")
plt.title("The Elbow method showing the optimal K value using Inertia value")
plt.show() 

#visulizing the Silhouette Score and k value
plt.plot(k,score,'bx-')
plt.xlabel("K value")
plt.ylabel("Silhouette Score")
plt.title("The Elbow method showing the optimal K value using Silhouette score")
plt.show() 

#applying K mean using 3 clusters
kmn=KMeans(n_clusters=6)
kmn.fit(dataset)
pred=kmn.predict(dataset)
sil_score=silhouette_score(dataset,pred)


#plotting the distribution
labels=kmn.labels_
X=dataset
dist = 1 - cosine_similarity(X)
pca = PCA(2)
pca.fit(dist)
X_PCA = pca.transform(dist)
X_PCA.shape
x, y = X_PCA[:, 0], X_PCA[:, 1]

colors = {0: 'red',
          1: 'blue',
          2: 'green', 
          3: 'yellow', 
          4: 'orange',  
          5:'purple'}
  
df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(20, 13)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
            color=colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.legend()
ax.set_title("Customers Segmentation based on their Credit Card usage bhaviour.")
plt.show()