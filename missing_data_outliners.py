import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

titanic = pd.read_csv('titanic.csv')


def test_data():
    print(titanic.head())
    # shape and statistical summary of dataset
    print(titanic.shape)
    print(titanic.describe())
    # check if certain row has null values
    print(titanic['Age'].isnull().sum())
    # check entire dataset
    print(titanic.isnull().sum())


def delete_missing_values():
    # delete any row with null values in the 'Embark' column
    global titanic
    print(f'Missing Embarked = {titanic["Embarked"].isnull().sum()}')
    titanic = titanic.dropna(subset=['Embarked'], axis=0)
    print(titanic['Embarked'].isnull().sum())


def imputing_missing_values():
    # replace any null values in 'Age' with the mean age
    delete_missing_values()
    global titanic
    mean_age = titanic['Age'].mean()
    print(f'Missing age = {titanic["Age"].isnull().sum()}')
    titanic['Age'] = titanic['Age'].fillna(mean_age)
    print(titanic['Age'].isnull().sum())

    # replace any null 'Cabin' with string missing
    print(f'Missing Cabin = {titanic["Cabin"].isnull().sum()}')
    titanic['Cabin'] = titanic['Cabin'].fillna('missing')
    print(titanic['Cabin'].isnull().sum())
    print(titanic.isnull().sum())


def outliers():
    global titanic
    # create a boxplot
    plt.figure(figsize=(10, 5))
    plt.boxplot(titanic['Fare'], vert=False)  # vert turns the plot horizontal or vertical
    plt.xlabel('Fare')
    plt.show()


def removing_outliers():
    # Find the index of the rows with the max fare value
    imputing_missing_values()
    global titanic
    max_fare = max(titanic['Fare'])
    max_fare_idx = titanic.index[titanic['Fare'] == max_fare].tolist()
    print(f'Passengers in rows {max_fare_idx} paid the highest fare of ${max_fare}')

    # removing the outliers
    titanic2 = titanic.drop(index=max_fare_idx)
    print(titanic2.describe())


def removing_outliers_q3():
    # Find the index of rows with fares greater than the third quartile
    imputing_missing_values()
    q3 = np.percentile(titanic['Fare'], 75)
    greater_than_q3_index = titanic.index[titanic['Fare'] > q3].tolist()
    print(f'{len(greater_than_q3_index)} passengers paid a fare greater than ${q3}')

    titanic3 = titanic.drop(index=greater_than_q3_index)
    print(titanic3.describe())


removing_outliers_q3()
