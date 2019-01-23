import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

auto = pd.read_csv('auto.csv')
print(auto.head())


def scatter_plot():
    # create a scatter plot to show relationship btwn horsepower and mpg
    plt.figure(figsize=(12, 6))
    plt.plot(auto['horsepower'], auto['mpg'], 'ro')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG (Miles Per Gallon)')

    plt.show()


# scatter_plot()

def linear_test():
    # initialize and fit linear regression model
    global auto
    x = auto['horsepower']
    y = auto['mpg']
    lm = LinearRegression()
    lm.fit(x[:, np.newaxis], y)

    print(f'alpha = {lm.intercept_}')
    print(f'betas = {lm.coef_}')

    # Create scatter plot to show relationship between horsepower and mpg and predict
    plt.figure(figsize=(12, 6))
    plt.plot(auto['horsepower'], auto['mpg'], 'ro')
    plt.plot(x, lm.predict(x[:, np.newaxis]))
    plt.xlabel('Horsepower')
    plt.ylabel('MPG (Miles Per Gallon)')

    plt.show()



linear_test()
