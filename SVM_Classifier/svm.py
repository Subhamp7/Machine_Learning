# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:37:42 2020

@author: subham
"""
#loading the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


#loading the dataset
data=pd.read_csv("glass_data.csv")

#checking the count of missing data
print("The count of missing data is :",data.isnull().sum().sum())

#the target output
print("The outputs are:\n",data["Type"].unique())

#plotting corelations
#sns.heatmap(data.corr(),annot=True)

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')

plt.xlim(-1, 3.5);