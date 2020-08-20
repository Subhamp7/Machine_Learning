# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics  import r2_score

# Importing the dataset
df = pd.read_csv('https://query.data.world/s/rjfb64km2adpnnrglujstxylw3wanb',nrows=5000)

#extracting the required data
data=df[['experience_total','salary']]

#checking for missing data
print("The missing data \n {} ".format(data.isnull().sum()))

#droping the rows with missing data
data=data.dropna( how='any')

#Splitting X and Y sets
X=data.iloc[:,0:1].values
Y=data.iloc[:,1:2].values

#splitting data into train and test sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.3, random_state=0)

#applying DT regressor
dt=DecisionTreeRegressor(max_depth=4,random_state = 0)
dt.fit(X_train,Y_train)

#plotting the training results
X_grid = np.arange(min(X_train), max(X_train), 0.01) 
X_grid = X_grid.reshape((len(X_grid), 1))  
plt.scatter(X_train,Y_train, color='red')
plt.plot(X_grid,dt.predict(X_grid),color='blue')
plt.title('DT')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#accuracy
print("The accuracy of the model is : ",r2_score(Y_test, dt.predict(X_test)))

#plotting the tree structure
plot_tree(dt,filled=True)