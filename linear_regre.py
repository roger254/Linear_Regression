import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(0)


def random_data():
    # Generate random data
    # array of  100 values with mean = 1.5 , standard deviation = 2.5
    x = 2.5 * np.random.randn(100) + 1.5
    # Prediction of Y, assuming a = 2, b = 0.3
    y_predicate = 2 + 0.3 * x
    # Generate 100 residual terms
    res = 0.5 * np.random.randn(100)
    # Actual values of Y
    y_actual = 2 + 0.3 * x + res

    df = pd.DataFrame(
        {
            'X': x,
            'y_predicate': y_predicate,
            'y_actual': y_actual
        }
    )
    return df, x, y_predicate, y_actual


def plot(df):
    # plot prediction as blue line, actual values of Y as red markers
    x = df['X']
    y_predicate = df['y_predicate']
    y_actual = df['y_actual']
    plt.figure(figsize=(12, 6))
    plt.plot(x, y_predicate)
    plt.plot(x, y_actual, 'ro')
    plt.title('Actual vs Predicated values from the dummy dataset')
    plt.show()


def r_square():
    # Show the first five rows of the dataframe
    df, x, y_predicate, y_actual = random_data()
    df.head()

    # plot
    plot(df)
    # calculate the mean of Y
    y_mean = np.mean(y_actual)
    print(f'Mean of Y = {y_mean}')

    # Calculate SSR and SST
    df['SSR'] = (df['y_predicate'] - y_mean) ** 2
    df['SST'] = (df['y_actual'] - y_mean) ** 2
    ssr = df['SSR'].sum()
    sst = df['SST'].sum()

    # Calculate R-squared
    r2 = ssr / sst
    print(f'R2 = {r2}')


def least_squares():
    # Calculate the mean of X and Y
    df, x, y_predicate, y_actual = random_data()
    x_mean = np.mean(x)
    y_mean = np.mean(y_actual)

    # calculate the terms needed for the numerator and denominator
    df['xy_cov'] = (df['X'] - x_mean) * (df['y_actual'] - y_mean)
    df['x_var'] = (df['X'] - x_mean) ** 2

    # calculate beta and alpha
    beta = df['xy_cov'].sum() / df['x_var'].sum()
    alpha = y_mean - (beta * x_mean)
    print(f'alpha = {alpha} \n beta = {beta}')

    # Store predictions
    df['y_predicate'] = alpha + beta * df['X']
    plot(df)
    return df, alpha, beta, y_mean


def improved_r_squares():
    df, alpha, beta, y_mean = least_squares()

    df['y_predicate'] = alpha + beta * df['X']

    # calculate the new SSR
    df['SSR'] = (df['y_predicate'] - y_mean) ** 2
    df['SST'] = (df['y_actual'] - y_mean) ** 2
    ssr = df['SSR'].sum()
    sst = df['SST'].sum()

    # calculate R2
    r2 = ssr / sst
    print(f'R2 = {r2}')
    # plot table

    plot(df)


def residual_standard_error():
    df, alpha, beta, y_mean = least_squares()
    df['SSD'] = (df['y_actual'] - df['y_predicate']) ** 2
    ssd = df['SSD'].sum()
    # calc RSE
    rse = np.sqrt(ssd / 98)  # n = 100
    print(f'RSE = {rse}')
    error = rse / y_mean
    print(f'Mean Y = {y_mean}.')
    print(f'Error = {error}.')


least_squares()
