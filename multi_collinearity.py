import pandas as pd
import statsmodels.formula.api as smf

advert = pd.read_csv('advertising.csv')

print(advert.corr())


def news_tv_radio():
    # initialize and fit model with newspaper as linear function of TV and Radio
    model = smf.ols('Newspaper ~ TV + Radio', data=advert).fit()

    # calculate VIF
    r2 = model.rsquared
    vif = 1 / (1 - r2)
    print(f'VIF for newspaper = {vif}')


def tv_news_radio():
    # Initialise and fit model with TV as a linear function of TV and Radio
    model = smf.ols('TV ~ Newspaper + Radio', data=advert).fit()

    # calculate vif
    r2 = model.rsquared
    vif = 1 / (1 - r2)
    print(f'VIF for tv = {vif}')


def radio_news_tv():
    # Initialise and fit model with Radio as a linear function of TV and Newspaper
    model = smf.ols('Radio ~ Newspaper + TV', data=advert).fit()

    # calculate vif
    r2 = model.rsquared
    vif = 1 / (1 - r2)
    print(f'VIF for radio = {vif}')


tv_news_radio()
news_tv_radio()
radio_news_tv()
