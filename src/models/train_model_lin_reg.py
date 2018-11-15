"""
Linear Regression Model for Predicting Final Score
"""
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
# import scipy.stats as scs
import matplotlib.pyplot as plt
%matplotlib inline

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


######################################################################

if __name__ == '__main__':

    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_train = y_train['estimated_final_score']
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    y_test = y_test['estimated_final_score']

    numeric_cols = ['num_of_prev_attempts', 'studied_credits', 'clicks_per_day', 'pct_days_vle_accessed','max_clicks_one_day','first_date_vle_accessed', 'avg_score', 'avg_days_sub_early','days_early_first_assessment', 'score_first_assessment']

    # fill and scale
    X_train.fillna(value = 0, inplace = True)
    y_train.fillna(value = 0, inplace = True)
    X_train = scale_subset(X_train, numeric_cols)
    X_train = sm.add_constant(X_train)
    X_test.fillna(value = 0, inplace = True)
    y_test.fillna(value = 0, inplace = True)
    X_test = scale_subset(X_test, numeric_cols)
    X_test = sm.add_constant(X_test)

    # estimator
    lin_reg = sm.OLS(y_train, X_train)

    lin_reg = lin_reg.fit()

    lin_reg.summary()

    y_test.shape
    predictions = lin_reg.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(rmse)
    explained_variance_score(y_test, predictions)
    r2_score(y_test, predictions)

    residuals = y_test - predictions

    plt.figure(figsize=(12,10))
    plt.scatter(x=residuals, y=y_test)

    plt.hist(residuals, bins=100)



    # # save model
    # pickle.dump(log_reg_model, open('models/logistic_regression_completion.p', 'wb'))

    # # evaluation
    # predictions = log_reg_model.predict(X_test)
    # roc_auc = roc_auc_score(y_test, predictions)
    # probas = log_reg_model.predict_proba(X_test)[:, 1:]
    # tprs, fprs, thresh = roc_curve(y_test, probas)
    # recall = recall_score(y_test, predictions)

    # print('Best Model: {}'.format(log_reg_model))
    # print('Best Model parameters: {}'.format(lr_clf.best_params_))
    # print('Best Model Log Loss: {}'.format(lr_clf.best_score_))
    # print('Roc Auc: {}'.format(roc_auc))
    # print('Recall Score: {}'.format(recall))
