
""" 
Functions -based solution
"""
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, confusion_matrix, recall_score, roc_auc_score, roc_curve, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


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


def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])

######################################################################

if __name__ == '__main__':

    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_train = y_train['module_not_completed']
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    y_test = y_test['module_not_completed']
    
    numeric_cols = ['num_of_prev_attempts', 'studied_credits',
    'clicks_per_day', 'pct_days_vle_accessed','max_clicks_one_day',
    'first_date_vle_accessed', 'avg_score', 'avg_days_sub_early',   'days_early_first_assessment',
    'score_first_assessment']
    # fill and scale
    X_train.fillna(value = 0, inplace = True)
    X_train = scale_subset(X_train, numeric_cols)
    X_test.fillna(value = 0, inplace = True)
    X_test = scale_subset(X_test, numeric_cols)

    # estimators
    lr = LogisticRegression()
    
    # GridSearch parameters
    lr_params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['newton-cg','lbfgs', 'liblinear'],
            'max_iter': [25, 50, 100, 200, 500, 1000],
            'warm_start': ['False', 'True'],
    }

    lr_clf = GridSearchCV(lr, param_grid=lr_params,
                        scoring='neg_log_loss',
                        n_jobs=-1,
                        cv=5)

    lr_clf.fit(X_train, y_train)

    log_reg_model = lr_clf.best_estimator_

    # save model
    pickle.dump(log_reg_model, open('models/logistic_regression_completion.p', 'wb'))

    # evaluation
    predictions = log_reg_model.predict(X_test)
    roc_auc = roc_auc_score(y_test, predictions)
    probas = log_reg_model.predict_proba(X_test)[:, 1:]
    tprs, fprs, thresh = roc_curve(y_test, probas)
    recall = recall_score(y_test, predictions)

    print('Best Model: {}'.format(log_reg_model))
    print('Best Model parameters: {}'.format(lr_clf.best_params_))
    print('Best Model Log Loss: {}'.format(lr_clf.best_score_))
    print('Roc Auc: {}'.format(roc_auc))
    print('Recall Score: {}'.format(recall))

    # plt.figure(figsize=(12,10))
    # plt.plot(fprs, tprs, 
    #      label='Logistic Regression', 
    #      color='red')
    # plt.plot([0,1],[0,1], 'k:')
    # plt.legend()
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    # plt.title("ROC Curve AUC: {} Recall: {}".format(roc_auc, recall))
    # plt.show()
