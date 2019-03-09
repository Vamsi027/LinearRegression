# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('')#Enter the csv file you want to import
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,-1].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_Y = StandardScaler()
Y = np.array(Y).reshape(-1,1)
Y = sc_Y.fit_transform(Y)
Y = Y.flatten()

#Fitting regression model to the dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,Y)

#Predicting a new result
y_pred=sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[]))))

#Visualizing SVR Results
plt.scatter(X,Y,color='')
plt.plot(X,regressor.predict(X),color='')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.show()
