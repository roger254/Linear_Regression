import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

advert = pd.read_csv('advertising.csv')

# Build linear regression model using TV and Radio as predictors
# Split data into predictors X and output Y
predictors = ['TV', 'Radio']
x = advert[predictors]
y = advert['Sales']

# split data into training and testing sets
trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=0)  # 80:20
print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

# initialize and fit model
lm = LinearRegression()
lm.fit(trainX, trainY)
# parameters of the model
print(f'alpha = {lm.intercept_}')
print(f'betas = {lm.coef_}')
print(f'R_squared = {lm.score(trainX, trainY)}')
# sales predict
print(lm.predict(testX))

# Recursive Feature Elimination
# Support Vector Regression
# start with all possible predictors
predictors = ['TV', 'Radio', 'Newspaper']
x = advert[predictors]
y = advert['Sales']

# estimate a linear model
estimator = SVR(kernel='linear')

# using rfe, specify 2 predictors for the final model
# and 1 predictor to remove at each iteration
selector = RFE(estimator, 2, step=1)
selector = selector.fit(x, y)
# get list of selected variables (TV, Radio, Newspaper)
print(selector.support_)
# selected variables have a ranking of 1
print(selector.ranking_)
