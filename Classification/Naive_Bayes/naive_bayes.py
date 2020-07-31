# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:22:08 2020

@author: subham
"""

#importing required libraries
import nltk
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
import pandas as pd
import seaborn as sns
import numpy as np

#importing the dataset
data=pd.read_csv("spam_data.csv",encoding='latin-1')

#checking for the nan value and droping the columns
print("Checking the null value",data.isnull().sum())
data=data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)

#renaming the column and printing the spam and ham count
data=data.rename(columns={"v1":"type","v2":"text"})
print("The count of Spam and Ham :\n",data.groupby('type').count())

#creating a column which consist of the len of text msg
data['length']=data['text'].apply(len)

#plotting the test type and its length
#sns.countplot(x='length',hue='type', data=data)

#removing the punctuation from text
def punct(text):
    punct=""
    for index in text:
        if( index not in punctuation):
            punct+=index
    return punct

#removing stop words from text
def stpwrd(text):
    out=[]
    for index in text.split():
        if(index.lower() not in stopwords.words('english')):
            out.append(index)
    return out

data["text"]=data["text"].apply(punct)
data["text"]=data["text"].apply(stpwrd)


piplin=Pipeline([('cntvct',CountVectorizer(analyzer=punct)),
                 ('tfid',TfidfTransformer()),
                 ('naive',MultinomialNB())])

X=data.iloc[:,1].values
Y=data.iloc[:,0]
Y=np.array(Y.replace({'ham' : 1 , 'spam' : 0 }))
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


piplin.fit(x_train,y_train)
pred=piplin.predict(x_test)

report=classification_report(pred,y_test)
cm=confusion_matrix(pred,y_test)

