#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing pandas libraries
import pandas as pd
import numpy  as np
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

#ref https://pandas.pydata.org/pandas-docs/stable/pandas.pdf


# In[2]:


#creating a series with index as Date
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102',periods=6))
print("The index are {} and the value are {} ".format(s1.index,s1.values))


# In[3]:


#converting a dictionary to a pandas Series
dict1={"India":100,"Japan":200,"Dubai":300}
s2=pd.Series(dict1)
s2


# In[4]:


#converting dictionary into DataFrame
dict2={"One":pd.Series([1,2,3], index=["a","b","c"]),
          "Two":pd.Series([1,2,3,4], index=["a","b","c","d"])}
s3=pd.DataFrame(dict2)
s3


# In[5]:


#adding a column in DataFrame
s3["Three"]=pd.Series([10,20,30], index=["a","b","c"])
s3


# In[6]:


#importing csv files from same location
dataset1=pd.read_csv("property_data.csv", sep=",", delimiter=None, header="infer")
dataset1


# In[7]:


#displaying the shape and description of DataFrame
print("The Shape of the DataFrame is {} \n\nThe description are: \n \n{} ".format(dataset1.shape, dataset1.describe()))


# In[8]:


#converting DataFrame to json
dataset1.to_json(orient='index')#orient="columns","values","split"


# In[10]:


#transpose
dataset1.T


# In[38]:


#sorting by column name
dataset1.sort_values(by="ST_NAME",axis=0,ascending=False)


# In[12]:


#selecting the columns after conditions
dataset1[dataset1["NUM_BEDROOMS"]>2]


# In[13]:


#checking the values
dataset1["NUM_BEDROOMS"]>2


# In[19]:


#selecting those values of bedroomm which are available in bathroom
dataset1[dataset1["NUM_BEDROOMS"].isin(dataset1["NUM_BATH"])]


# In[22]:


#checking the null values
pd.isna(dataset1)


# In[20]:


#droping the value with missing data
dataset1.dropna(how="any")


# In[84]:


#filling the value with missing data with previous value
dataset1.ffill()#bfill() also can be used


# In[21]:


#filling the value with missing data with random values
dataset1.fillna(value=10)


# In[34]:


#grouping and count
dataset1.groupby("ST_NAME").sum()#any arithmetic operations


# In[41]:


#reading titanic dataset
dataset_Titanic=pd.read_csv("https://raw.githubusercontent.com/Subhamp7/ML_Kaggle/master/Titanic_Disaster_ML/train.csv")
dataset_Titanic.head()


# In[43]:


#checking the datatype
dataset_Titanic.dtypes


# In[48]:


#getting the type of single columns
type(dataset_Titanic["Name"])


# In[49]:


#using OR condition
dataset_Titanic[(dataset_Titanic["Pclass"] == 2) | (dataset_Titanic["Pclass"] == 3)]


# In[66]:


#count of Age column with nan value
dataset_Titanic["Age"].isna().value_counts()


# In[72]:


#using str splitting the strings column
dataset_Titanic["Name"].str.split().str.get(1)#str.get(1) to get first elememt of all rows


# In[76]:


#serching for particular field in column
dataset_Titanic["Name"].str.contains("Mr").value_counts()


# In[80]:


#replacing the values
dataset_Titanic["Sex"].replace({"male": "M","female": "F"})


# In[82]:


#to calculate the sum or any other arithmetic operations
dataset_Titanic["Fare"].agg(["sum","mean"])


# In[85]:


#to get the top 2 values
dataset_Titanic["Fare"].nlargest(2)


# In[ ]:




