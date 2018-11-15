"""
Linear Regression Model for Predicting Final Score
"""
import numpy as np
import pandas as pd
import pickle
import collections as c

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
# import scipy.stats as scs
import matplotlib.pyplot as plt

def scale_subset(df, columns):
    '''
    Use sklearn StandardScalar to scale only numeric columns.

    Parameters:
    ----------
    input {dataframe, list}: dataframe containing mixed feature variable types, list of names of numeric feature columns
    output: {dataframe}: dataframe with numeric features scaled and categorical features unchanged

    '''
    scalar = StandardScaler()
    numeric = df[columns]
    categorical = df.drop(columns, axis = 1)
    scalar.fit(numeric)
    num_scaled = pd.DataFrame(scalar.transform(numeric))
    num_scaled.rename(columns = dict(zip(num_scaled.columns, numeric_cols)), inplace = True)
    return pd.concat([num_scaled, categorical], axis = 1)


def only_completed(X_train, y_train, X_test, y_test, y_train_not_comp, y_test_not_comp):
    '''
    Returns dataframes with only those students who completed the course for the purpose of regressing the final score.
    '''
    test_indices = []
    train_indices = []

    y_test_not_comp = y_test_not_comp[y_test_not_comp['module_not_completed'] == 1]
    for index, row in y_test_not_comp.iterrows():
        test_indices.append(index)

    y_train_not_comp = y_train_not_comp[y_train_not_comp['module_not_completed'] == 1]
    for index, row in y_train_not_comp.iterrows():
        train_indices.append(index)

    return X_train.drop(train_indices), y_train.drop(train_indices), X_test.drop(test_indices), y_test.drop(test_indices)


######################################################################

if __name__ == '__main__':

    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_train_not_comp = y_train[['module_not_completed']]
    y_train = y_train['estimated_final_score']
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    y_test_not_comp = y_test[['module_not_completed']]
    y_test = y_test['estimated_final_score']

    numeric_cols = ['num_of_prev_attempts', 'studied_credits', 'clicks_per_day', 'pct_days_vle_accessed','max_clicks_one_day','first_date_vle_accessed', 'avg_score', 'avg_days_sub_early','days_early_first_assessment', 'score_first_assessment']

    # fill and scale
    X_train.fillna(value = 0, inplace = True)
    y_train.fillna(value = 0, inplace = True)
    X_train = scale_subset(X_train, numeric_cols)
    X_test.fillna(value = 0, inplace = True)
    y_test.fillna(value = 0, inplace = True)
    X_test = scale_subset(X_test, numeric_cols)

    # only students who completed the course
    X_train, y_train, X_test, y_test = only_completed(X_train, y_train, X_test, y_test, y_train_not_comp, y_test_not_comp)

    # estimator

    lr = LinearRegression(n_jobs=-1)

    lr.fit(X_train, y_train)

    # evaluation
    predictions = lr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(rmse)
    evs = explained_variance_score(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print('RMSE: {}'.format(rmse))
    print('R-squared: {}'.format(r2))

    residuals = y_test - predictions

    plt.figure(figsize=(12,10))
    plt.scatter(x=residuals, y=y_test, alpha = 0.1, c='green')
    plt.show()

    plt.figure(figsize=(12,10))
    plt.hist(residuals, bins=100)
    plt.show()

    plt.figure(figsize=(12,10))
    plt.hist(y_train, bins=100)
    plt.show()

    abs_coef = list(np.abs(lr.coef_.ravel()))
    features = list(X_test.columns)
    coef_dict = c.OrderedDict((zip(abs_coef, features)))
    sorted(coef_dict.items(), reverse=True)

    # save model
    pickle.dump(lr, open('models/linear_regression_score.p', 'wb'))
