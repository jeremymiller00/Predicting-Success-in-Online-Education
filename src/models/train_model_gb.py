
""" 
Functions -based solution
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, confusion_matrix, recall_score, roc_auc_score, roc_curve, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier()


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
    'first_date_vle_accessed', 'avg_score', 'avg_days_sub_early',  'days_early_first_assessment',
    'score_first_assessment']

    X_train.fillna(value = 0, inplace = True)

    X_test.fillna(value = 0, inplace = True)

    # estimator
    gb = GradientBoostingClassifier()
    
    # GridSearch parameters
    gb_params = {
            'max_depth': [2, 3, 5],
            'learning_rate': [0.001, 0.01, 0.1, 1],
            'n_estimators': [50, 100, 500, 1000, 5000],
            'subsample': [1, 0.5, 0.3, 0.1],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_impurity_decrease': [0, 1, 5],
    }
    
    gb_clf = GridSearchCV(gb, param_grid=gb_params,
                        scoring='recall',
                        n_jobs=-1,
                        cv=5)

    gb_clf.fit(X_train, y_train)

    gb_model = gb_clf.best_estimator_

    # save model
    pickle.dump(rf_model, open('src/models/completion_classifier_gb.p', 'wb')) 


    # evaluation
    predictions = gb_model.predict(X_test)
    roc_auc = roc_auc_score(y_test, predictions)
    probas = gb_model.predict_proba(X_test)[:, 1:]
    tprs, frps, thresh = roc_curve(y_test, probas)
    recall = recall_score(y_test, predictions)

    print('Best Model: {}'.format(rf_model))
    print('Best Model parameters: {}'.format(rf_clf.best_params_))
    print('Best Model Log Recall: {}'.format(recall))
    print('Best Model Roc Auc: {}'.format(roc_auc))

    # plt.figure(figsize=(12,10))
    # plt.plot(fprs, tprs, 
    #      label='Random Forest', 
    #      color='red')
    # plt.plot([0,1],[0,1], 'k:')
    # plt.legend()
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    # plt.title("ROC Curve AUC: {}".format(roc_auc))
    # plt.show()
