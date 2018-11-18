
""" 
Functions -based solution
"""
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, confusion_matrix, recall_score, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
# import statsmodels.api as sm
import matplotlib.pyplot as plt
import collections as c


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

def print_roc_curve(y_test, probabilities):
    '''
    Calculates and prints a ROC curve given a set of test classes and probabilities from a trained classifier
    '''
    tprs, fprs, thresh = roc_curve(y_test, probabilities)
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

######################################################################

if __name__ == '__main__':

    X_train = pd.read_csv('data/processed/first_half/X_train.csv')
    y_train = pd.read_csv('data/processed/first_half/y_train.csv')
    y_train = y_train['module_not_completed']
    X_test = pd.read_csv('data/processed/first_half/X_test.csv')
    y_test = pd.read_csv('data/processed/first_half/y_test.csv')
    y_test = y_test['module_not_completed']
    
    numeric_cols = ['num_of_prev_attempts', 'studied_credits', 'module_presentation_length', 'sum_click_dataplus', 'sum_click_dualpane', 'sum_click_externalquiz', 'sum_click_forumng','sum_click_glossary', 'sum_click_homepage', 'sum_click_htmlactivity', 'sum_click_oucollaborate', 'sum_click_oucontent', 'sum_click_ouelluminate', 'sum_click_ouwiki', 'sum_click_page', 'sum_click_questionnaire', 'sum_click_quiz', 'sum_click_repeatactivity', 'sum_click_resource', 'sum_click_sharedsubpage', 'sum_click_subpage', 'sum_click_url', 'sum_days_vle_accessed', 'max_clicks_one_day', 'first_date_vle_accessed', 'avg_score', 'avg_days_sub_early', 'days_early_first_assessment', 'score_first_assessment']

    # fill and scale
    X_train.fillna(value = 0, inplace = True)
    X_train = scale_subset(X_train, numeric_cols)
    X_test.fillna(value = 0, inplace = True)
    X_test = scale_subset(X_test, numeric_cols)

    # remove features with VIF > 10
    '''
    [(34.82408402588052, 'code_presentation_2014J'),
    (27.902556097941535, 'module_presentation_length'),
    (17.71032093366156, 'sum_days_vle_accessed'),
    (12.29325003243407, 'avg_score'),
    (12.129102421725705, 'code_module_BBB'),
    (11.26607390128323, 'score_first_assessment'),
    (10.74221505000013, 'code_module_DDD'),
    (10.185010067236496, 'code_module_FFF') '''

    high_vif = ['code_presentation_2014J', 'module_presentation_length', 'sum_days_vle_accessed', 'avg_score', 'code_module_BBB', 'score_first_assessment', 'code_module_DDD', 'code_module_FFF']
    X_train.drop(high_vif, axis = 1, inplace = True)
    X_test.drop(high_vif, axis = 1, inplace = True)
    
    # # estimators
    # lr = LogisticRegression()
    
    # # GridSearch parameters
    # lr_params = {
    #         'C': [0.01, 0.1, 1, 10, 100],
    #         'penalty': ['l2'],
    #         'tol': [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001],
    #         'solver': ['newton-cg','lbfgs', 'liblinear'],
    #         'max_iter': [10, 25, 50, 100, 200, 500],
    #         'warm_start': ['False', 'True'],
    # }

    # lr_clf = GridSearchCV(lr, param_grid=lr_params,
    #                     scoring='neg_log_loss',
    #                     n_jobs=-1,
    #                     cv=5)

    # lr_clf.fit(X_train, y_train)
    # log_reg_model = lr_clf.best_estimator_

    # best model as determined by grid search:
    log_reg_model = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=10, multi_class='warn', n_jobs=None, penalty='l2', random_state=None, solver='newton-cg', tol=1e-09, verbose=0, warm_start='False')

    cross_val_score(log_reg_model, X_train, y_train, scoring = 'roc_auc', cv=5)
    cross_val_score(log_reg_model, X_train, y_train, scoring = 'recall', cv=5)
    cross_val_score(log_reg_model, X_train, y_train, scoring = 'precision', cv=5)
    cross_val_score(log_reg_model, X_train, y_train, scoring = 'accuracy', cv=5)

    # save model
    pickle.dump(log_reg_model, open('models/logistic_regression_completion_first_half.p', 'wb'))

'''
    # evaluation
    predictions = log_reg_model.predict(X_test)
    roc_auc = roc_auc_score(y_test, predictions)
    probas = log_reg_model.predict_proba(X_test)[:, 1:]
    tprs, fprs, thresh = roc_curve(y_test, probas)
    recall = recall_score(y_test, predictions)
    conf_mat = standard_confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    print_roc_curve(y_test, probas)
    print('Best Model: {}'.format(log_reg_model))
    print('Best Model parameters: {}'.format(lr_clf.best_params_))
    print('Best Model Log Loss: {}'.format(lr_clf.best_score_))
    print('Roc Auc: {}'.format(roc_auc))
    print('Recall Score: {}'.format(recall))
    print('Confusion Matrix: {}'.format(conf_mat))
    print('Classification Report: {}'.format(class_report))

    # Feature Importances
    abs_coef = list(np.abs(log_reg_model.coef_.ravel()))
    features = list(X_test.columns)
    coef_dict = c.OrderedDict((zip(abs_coef, features)))
    ordered_feature_importances = sorted(coef_dict.items(), reverse=True)
    print('The top ten features affecting completion are: {}'.format( ordered_feature_importances[:10]))
'''