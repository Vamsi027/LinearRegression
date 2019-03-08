# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv()#Enter the csv file you want to import
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

#categorical values into digits or numbers
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy variable trap :- Although the library will take care of it.We can do it manually.
X=X[:,1:] 

#splitting into testset and trainingset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#predicting the test results
Y_pred=regressor.predict(X_test)

#Building an optimal model using Backward Elimination
import statsmodels.formula.api as sm  #let us consider P>0.05 shall be removed
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1) #axis=1 means column else if axis=0 means append a row
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
