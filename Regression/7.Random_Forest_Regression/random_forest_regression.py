# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 5, random_state = 0)
regressor.fit(X, Y)


# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()