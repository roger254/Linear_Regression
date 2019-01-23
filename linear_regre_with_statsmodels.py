import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf

# import display rows of advertising
advert = pd.read_csv('advertising.csv')
advert.head()


def plot(advert_):
    plt.figure(figsize=(12, 6))
    plt.plot(advert_['TV'], advert_['Sales'], 'o')
    plt.xlabel('TV Advertising Costs')
    plt.ylabel('Sales')
    plt.title('TV vs Sales')

    plt.show()


def line_of_best_fit(advert_):
    # initialize and fit linear regression model using stats_model
    model1 = smf.ols('Sales ~ TV', data=advert_)
    model1 = model1.fit()
    print(model1.params)
    print(model1.rsquared)
    print(model1.summary())

    # plot regression against actual data
    sales_pred = model1.predict(advert['TV'])
    advert['sales_pred'] = sales_pred
    plt.figure(figsize=(12, 6))
    plt.plot(advert['TV'], advert['Sales'], 'o')  # scatter plot showing actual data
    plt.plot(advert['TV'], sales_pred, 'r', linewidth=2)  # regression line
    plt.xlabel('TV Advertising Costs')
    plt.ylabel('Sales')
    plt.title('TV vs Sales')

    plt.show()
    print(model1.predict({'TV': 400}))
    return advert_


def calc_rse():
    advert_ = line_of_best_fit(advert)
    advert_['SSD'] = (advert_['Sales'] - advert_['sales_pred']) ** 2
    ssd = advert_['SSD'].sum()
    rse = np.sqrt(ssd / 198)  # n = 200
    sales_mean = np.mean(advert_['Sales'])
    error = rse / sales_mean

    print(f'RSE = {rse} \nMean sale = {sales_mean} \nError ={np.round(error, 4) * 100}%')
    print(f"Accuracy = {100 - np.round(error, 4) * 100}")


line_of_best_fit(advert)
