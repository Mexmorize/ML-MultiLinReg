# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:32:20 2019

@author: Benjamin B
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Gives each state a number (0,1,2,3,etc.) aka: text to numbers
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])

# Give each state it's respective column to be either 0 or 1
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# Avoiding Dummy Variable Trap
x = x[:, 1:]

# Splitting the dataset into Training and Testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test Set results
y_pred = regressor.predict(x_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
"""Adds a column of 1's in position x[0] to unify multiple linear regression
formula instead of having b[0] at the start, now it's 
b[0]*x[0] + ... b[n]*x[n]"""
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)

# Actually starting Backward Elimination
"""Create optimal model of x with only statistical significant 
independent variables. Start with original All-in model and remove variables
one by one. Create x_opt to have all the rows and all the columns of x"""
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
"""Make a regressor for the ordinary least squares model.
endog is the dependent variable (y), exog is the array of observations
(x_opt) NOTE: the intercept is not included  by default, that's why we 
needed to do the thing above ^"""
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
"""Summary method gives a lot of information about the data"""
regressor_OLS.summary()

"""Found that the dummy variable for state in x[2] slot had a p-value of 
0.990 so that is above the 0.05 SL so we remove it and re-fit"""
x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

"""Found that the dummy variable for state in x[1] slot had a p-value of 
0.940 so that is above the 0.05 SL so we remove it and re-fit"""
x_opt = x[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

"""Found that the dummy variable for state in x[2] slot had a p-value of 
0.602 so that is above the 0.05 SL so we remove it and re-fit"""
x_opt = x[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

"""Found that the dummy variable for state in x[2] slot had a p-value of 
0.060 so that is above the 0.05 SL so we CAN DECIDE TO remove it 
and re-fit, although then the model would only be a single Linear
Regression model and only take into account the original x[3] column
which would only correlate R&D spending to profit amount. I chose to
keep the value even though it's 0.060 since it is very close to the SL
of 0.05"""
# x_opt = x[:, [0, 3]]
# regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
# regressor_OLS.summary()


"""
Hi guys,

if you are also interested in some automatic implementations of Backward Elimination in Python, please find two of them below:

Backward Elimination with p-values only:

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)



Backward Elimination with p-values and Adjusted R Squared:

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

"""