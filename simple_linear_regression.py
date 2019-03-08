# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv()#Enter your csv file
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

#splitting into testset and trainingset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#fitting simple linear regression to training sets
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#predicting the test set
y_pred=regressor.predict(X_test)

#visualizing the training results
plt.scatter(X_train,Y_train ,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

#visualizing the test results
plt.scatter(X_test,Y_test ,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

#testing the goodness of fit
from sklearn.metrics import mean_squared_error
r2_score=regressor.score(X_train,Y_train)
