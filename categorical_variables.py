import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Age (X1)': [42, 24, 47, 50, 60],
    'Monthly Income (X2)': [7313, 17747, 22845, 18552, 14439],
    'Gender (X3)': ['Female', 'Female', 'Male', 'Female', 'Male'],
    'Total Spend (Y)': [4198.385084, 4134.976648, 5166.614455, 7784.447676, 3254.160485]
}
df = pd.DataFrame.from_dict(data=data)
print(df)

# create a dummy set
dummy_gender = pd.get_dummies(df['Gender (X3)'], prefix='Sex')
print(dummy_gender)

# append dummy data and drop Gender
df_new = df.join(dummy_gender).drop(['Gender (X3)'], 1)
print(df_new)

# create a linear regression model using scikit-learn
predictors = ['Age (X1)', 'Monthly Income (X2)', 'Sex_Female', 'Sex_Male']
X = df_new[predictors]
Y = df_new['Total Spend (Y)']
lm = LinearRegression()
lm.fit(X, Y)
print(f'alpha = {lm.intercept_}')
print(f'betas = {lm.coef_}')
print(f'R2 = {lm.score(X, Y)}')
