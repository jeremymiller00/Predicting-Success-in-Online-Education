'''
For when you get a new datapoint
'''

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score


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

    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    y_test = y_test['module_not_completed']

    numeric_cols = ['num_of_prev_attempts', 'studied_credits',
    'clicks_per_day', 'pct_days_vle_accessed','max_clicks_one_day',
    'first_date_vle_accessed', 'avg_score', 'avg_days_sub_early',  'days_early_first_assessment',
    'score_first_assessment']

    # fill and scale
    X_test.drop(['date_unregistration', 'id_student'], axis = 1, inplace = True)
    X_test.fillna(value = 0, inplace = True)
    X_test = scale_subset(X_test, numeric_cols)

    # load the model
    model = pickle.load(open('models/logistic_regression_completion.p', 'rb'))

    pred = model.predict(X_test) 

    # evaluation
    predictions = log_reg_model.predict(X_test)
    roc_auc = roc_auc_score(y_test, predictions)
    probas = log_reg_model.predict_proba(X_test)[:, 1:]
    tprs, fprs, thresh = roc_curve(y_test, probas)
    recall = recall_score(y_test, predictions)

    plt.figure(figsize=(12,10))
    plt.plot(fprs, tprs, 
         label='Logistic Regression', 
         color='red')
    plt.plot([0,1],[0,1], 'k:')
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve AUC: {} Recall: {}".format(roc_auc, recall))
    plt.show()
