# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:49:42 2020

@author: subham
"""

import numpy as np
import pandas as pd
import os
from os import path

if(path.exists("wine.data")):
    my_file = 'wine.data'
    base = os.path.splitext(my_file)[0]
    os.rename(my_file, base + '.csv')

dataset=pd.read_csv("wine.csv", header=None)
X=dataset.iloc[:,1:]
Y=dataset.iloc[:,0]

from sklearn.model_selection import train_test_split
X_train, Y_train, X_test, Y_test =train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
Y_train=sc.transform(Y_train)









