# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:35:07 2020

@author: subham
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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

