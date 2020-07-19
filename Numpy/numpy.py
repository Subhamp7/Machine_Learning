#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing numpy
import numpy as np


# In[9]:


#converting a list into array
list_1=[3,7,8,0,1,2,5]
array=np.array(list_1)
print("The array is :",array)
print("The data type of array is :",type(array))
print("The shape of array is :",array.shape)


# In[20]:


#merging many list into one array and reshaping
List_1=[2,6,9,3,5,0]
List_2=[9,4,7,0,1,2]
array_2=np.array([List_1,List_2])
print("The array is :\n",array_2)

array_2=array_2.reshape(3,4)#should be of same size as the original array
print("\nThe array after reshaping is :\n",array_2)


# In[29]:


#indexing in array
print("The first row\n",array_2[0])
print("\nFirst element if second row",array_2[1][0])
print("\nSelecting a shape of array\n",array_2[0:2,1:])


# In[41]:


#generating array
array_3=np.arange(1,15,2)#start,stop,interval
print("The array is :",array_3)

array_4=np.linspace(2,90,15)
print("The array is :\n",array_4)


# In[50]:


#brodcasting array
array_5=np.arange(1,10)
array_5[4:]=5
print("The updated array is :",array_5)


# In[62]:


#arithmetic operations
array_6=np.arange(50,100,5).reshape(2,5)
print("The array after adding 2 :\n",array_6+2)

print("\nPrinting the array which is higher than 60 :\n",array_6[array_6>60])

print("\nApplying conditional statement :\n",array_6>60)


# In[69]:


#generating array with ones and zeros
array_7=np.ones((2,4),dtype=int)#zeros for zero
print("The arrays with ones :\n",array_7)


# In[75]:


#generating the array with random distribution
array_8=np.random.rand(2,3)
print("The random values are :\n",array_8)


# In[78]:


#generating the array with standard random distribution
array_8=np.random.randn(2,3)
print("The random values are :\n",array_8)

#visualizing the standard random distributed array
import seaborn as sns
sns.distplot(array_8)


# In[ ]:




