# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import statsmodels.api as sm

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#printing the slope and intercept
print("Intercept:" ,regressor.intercept_)
print("Slope:",regressor.coef_)

#passing the X_test value to predict 
pred=regressor.predict(X_test)

# Evaluating Model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print("Accuracy:",metrics.r2_score(y_test, pred))


X = sm.add_constant(X)
est = sm.OLS(y, X)
est2 = est.fit()
print("summary()\n",est2.summary())
print("------------------------------------------------------------------\n")
print("pvalues\n",est2.pvalues)
print("------------------------------------------------------------------\n")
print("tvalues\n",est2.tvalues)
print("------------------------------------------------------------------\n")
print("rsquared\n",est2.rsquared)
print("------------------------------------------------------------------\n")
print("rsquared_adj\n",est2.rsquared_adj)


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

