
""" 
Functions -based solution
"""
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, confusion_matrix, recall_score, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
    
    # # estimators
    svc = SVC()
    
    # GridSearch parameters
    svc_params = {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear'],
            'tol': [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001],
            'probability': ['True'],
            'max_iter': [10, 25, 50, 100, 200, 500],
    }

    svc_clf = GridSearchCV(svc, param_grid=svc_params,
                        scoring='roc_auc',
                        n_jobs=-1,
                        cv=5)

    svc_clf.fit(X_train, y_train)
    svc_model = svc_clf.best_estimator_

    # best model as determined by grid search:
    # svc_model = 

    # evaluation
    roc_auc_cv = (cross_val_score(svc_model, X_train, y_train, scoring = 'roc_auc', cv=5))
    recall_cv = cross_val_score(svc_model, X_train, y_train, scoring = 'recall', cv=5)
    precision_cv = cross_val_score(svc_model, X_train, y_train, scoring = 'precision', cv=5)
    accuracy_cv = cross_val_score(svc_model, X_train, y_train, scoring = 'accuracy', cv=5)
    f1_cv = cross_val_score(svc_model, X_train, y_train, scoring = 'f1_micro', cv=5)

    print('Best Model: {}'.format(svc_model))
    print('Roc Auc: {}'.format(roc_auc_cv))
    print('Recall Score: {}'.format(recall_cv))
    print('Precision Score: {}'.format(precision_cv))
    print('Accuracy Score: {}'.format(accuracy_cv))
    print('F1 Micro: {}'.format(f1_cv))

    # save model
    pickle.dump(svc_model, open('models/svc_completion_first_half.p', 'wb'))

    # assessing variance inflation
    # vif = []
    # for v in range(len(X_train.columns)):
    #     vif.append(variance_inflation_factor(X_train.values, v))
    # features = list(X_test.columns)
    # vif_dict = c.OrderedDict((zip(vif, features)))
    # sorted(vif_dict.items(), reverse=True)

    # feature correlation
    # cor = X_train.corr().abs()
    # s = cor.unstack()
    # so = s.sort_values(kind="quicksort", ascending=False)
    # so[58::2]

'''
    # final model evaluation (see jupyter notebook)
    predictions = svc_model.predict(X_test)
    roc_auc = roc_auc_score(y_test, predictions)
    probas = svc_model.predict_proba(X_test)[:, :1]
    tprs, fprs, thresh = roc_curve(y_test, probas)
    recall = recall_score(y_test, predictions)
    conf_mat = standard_confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    print_roc_curve(y_test, probas)
    print('Best Model: {}'.format(svc_model))
    print('\nRoc Auc: {}'.format(roc_auc))
    print('\nRecall Score: {}'.format(recall))
    print('\nClassification Report:\n {}'.format(class_report))
    print('\nConfusion Matrix:\n {}'.format(standard_confusion_matrix(y_test, predictions)))

    # Feature Importances
    abs_coef = list(np.abs(svc_model.coef_.ravel()))
    features = list(X_test.columns)
    coef_dict = c.OrderedDict((zip(abs_coef, features)))
    print('The top ten features affecting completion are:\n')
    sorted(coef_dict.items(), reverse=True)[:10]

'''
