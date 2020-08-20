# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics  import r2_score

# Importing the dataset
# Importing the dataset
df = pd.read_csv('https://query.data.world/s/rjfb64km2adpnnrglujstxylw3wanb',nrows=500)

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

# Training the Random Forest Regression model on the whole dataset
regressor = RandomForestRegressor(n_estimators = 2, random_state = 0)
regressor.fit(X_train, Y_train)

#accuracy
print("The accuracy of the model is : ",r2_score(Y_test, regressor.predict(X_test)))

# Visualising the Random Forest Regression results (higher resolution)
X_grid = (np.arange(min(X_train), max(X_train), 0.01)).reshape(-1, 1)
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Random Forest Regression Train')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Random Forest Regression results (higher resolution)
X_grid = (np.arange(min(X_test), max(X_test), 0.01)).reshape(-1, 1)
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Random Forest Regression Test')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()