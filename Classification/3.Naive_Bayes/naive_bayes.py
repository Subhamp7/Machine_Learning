# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:22:08 2020

@author: subham
"""

#importing required libraries
import re
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import metrics

#importing the dataset
data=pd.read_csv("spam_ham_dataset.csv",encoding='latin-1')

#checking for the nan value and droping the columns
print("Checking the null value",data.isnull().sum())
data=data.drop(['Unnamed: 0','label'],axis=1)

# printing the spam and ham count
print("The count of Spam and Ham :\n",data.groupby('label_num').count())

#creating a column which consist of the len of text msg
data['length']=data['text'].apply(len)

#plotting the test type and its length
sns.countplot(x='length',hue='label_num', data=data)

#stemming and reoving stop words
ps = PorterStemmer()
corpus = []
for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#applying tfidvectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,5))
X=tfidf_v.fit_transform(corpus).toarray()
Y=data.iloc[:,1]

#splitting the dataset into test and train
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

#hyperparameter tunning
previous_score=0
alpha_value=[]
for alpha in np.arange(0,1,0.01):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(x_train,y_train)
    y_pred=sub_classifier.predict(x_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    alpha_value.append({alpha:score})
    
#applying naive bayes classifier
nb=MultinomialNB(alpha=0.01)
nb.fit(x_train,y_train)

#predicting the test data
pred=nb.predict(x_test)
      
#calculating the metrics
report=classification_report(pred,y_test)
cm=confusion_matrix(pred,y_test)

