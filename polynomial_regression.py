# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

#Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor=PolynomialFeatures(degree=4)
X_poly=polynomial_regressor.fit_transform(X)
linear_regressor2=LinearRegression()
linear_regressor2.fit(X_poly,Y)

#Visualizing the dataset on Polynomial Regression
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,linear_regressor2.predict(polynomial_regressor.fit_transform(X_grid)),color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
