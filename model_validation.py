import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

advert = pd.read_csv('advertising.csv')

# split dataset (80:20)
training = advert.sample(frac=0.8, random_state=0)
testing = advert.drop(training.index)

print(f"Number of training rows: {len(training)}")
print(f'Number of testing rows: {len(testing)}')

# Fit model with TV and Radio as predictors to training data
model = smf.ols('Sales ~ TV + Radio', data=training).fit()
print(model.summary())

# sales predict
training['sales_pred'] = model.predict()
print(training.head(20))

# store parameter values
alpha = model.params[0]
beta1 = model.params[1]
beta2 = model.params[2]

# calculate RSE for training set
training['SSD'] = (training['Sales'] - training['sales_pred']) ** 2
ssd = training['SSD'].sum()
rse = np.sqrt(ssd / 157)  # n = 160, p = 2
sales_mean = np.mean(training['Sales'])
error = rse / sales_mean
print(f'(Training) RSE = {rse}\nMean sale = {sales_mean}\nError = {np.round(error, 4) * 100}%')
print(f'(Training) Accuracy = {100 - np.round(error, 4) * 100}')

# calculate RSE for testing set
testing['sales_pred'] = alpha + (beta1 * testing['TV']) + (beta2 * testing['Radio'])
testing['SSD'] = (testing['Sales'] - testing['sales_pred']) ** 2
ssd = testing['SSD'].sum()
rse = np.sqrt(ssd / 37)  # n = 40, p = 2
sales_mean = np.mean(testing['Sales'])
error = rse / sales_mean
print(f'(Testing) RSE = {rse}\nMean sale = {sales_mean}\nError = {np.round(error, 4) * 100}%')
print(f'(Testing) Accuracy = {100 - np.round(error, 4) * 100}')
