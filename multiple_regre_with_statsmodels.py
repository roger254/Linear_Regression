import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

advert = pd.read_csv('advertising.csv')
advert.head()


def tv_newspaper():
    # initialize and fit new model with TV and Newspaper
    model = smf.ols('Sales ~ TV + Newspaper', data=advert).fit()
    print(model.summary())
    # equation => Sales = 5.77 + 0.046 * TV + 0.044 * Newspaper

    # store the parameters
    alpha = model.params[0]
    beta1 = model.params[1]
    beta2 = model.params[2]
    rse(model)


def rse(model, p=2):
    # calculate rse
    advert['sales_pred'] = model.predict()
    advert['SSD'] = (advert['Sales'] - advert['sales_pred']) ** 2
    ssd = advert['SSD'].sum()
    d = 200 - (p + 1)
    rse = np.sqrt(ssd / d)  # n =200, p =2
    sales_mean = np.mean(advert['Sales'])
    error = rse / sales_mean
    print(f'RSE = {rse} \nMean sale = {sales_mean} \nError = {np.round(error, 4) * 100}%')
    print(f"Accuracy = {100 - np.round(error, 4) * 100}")


def tv_radio():
    model = smf.ols('Sales ~ TV + Radio', data=advert).fit()
    print(model.summary())
    # equation => Sales = 2.92 + 0.045 * TV + 1.880 * Radio
    rse(model)


def tv_radio_newspaper():
    model = smf.ols('Sales ~ TV + Radio + Newspaper', data=advert).fit()
    print(model.summary())
    # equation => Sales = 2.939 + 0.046 * TV + 0.1885 * Radio + -0.001 * Newspaper
    rse(model, p=3)


# tv_newspaper()
# tv_radio()
tv_radio_newspaper()
