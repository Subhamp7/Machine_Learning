# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:06:40 2020

@author: subham
"""

#importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
from sklearn.metrics import silhouette_score

#loading the dataset
dataset=pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv")
details=dataset.describe()

#visualizing the quantity of categories
sns.countplot(y='species',data=dataset)#y for horizontal and vice versa

#comparing columns with species columns
list_plot=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for index in list_plot:
    sns.boxplot(index,'species',data=dataset)

#encoding the species columns
encoder=LabelEncoder()
dataset["species"]=encoder.fit_transform(dataset["species"])

sns.jointplot(x='sepal_length',y='species',data=dataset)

#applying K mean using 5 clusters
kmn=KMeans(n_clusters=3)
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



plt.plot(inertia,'bx-')
plt.set_xlabel("K value")
plt.set_ylabel("Inertia")
plt.set_title("The Elbow method showing the optimal K value using Inertia value")


plt.plot(k,distortions,'bx-')
plt.set_xlabel("K value")
plt.set_ylabel("Silhouette Score")
plt.set_title("The Elbow method showing the optimal K value using Silhouette score")

plt.show() 