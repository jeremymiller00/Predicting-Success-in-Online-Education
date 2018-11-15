
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
from sklearn.ensemble import RandomForestClassifier


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

def dropcol_importances(rf, X_train, y_train):
    '''
    Calculates the drop-column feature importances of a Random Forest model. Explanation here: https://explained.ai/rf-importance/index.html


    '''

    rf_ = clone(rf)
    rf_.random_state = 999
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
            data={'Feature':X_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I

######################################################################

if __name__ == '__main__':

    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_train = y_train['module_not_completed']
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    y_test = y_test['module_not_completed']

    X_train.fillna(value = 0, inplace = True)

    X_test.fillna(value = 0, inplace = True)

    # estimator
    rf = RandomForestClassifier()
    
    # GridSearch parameters
    rf_params = {
        'n_estimators': [50, 100, 1000], 
        'max_depth': [5, 10, 50], 
        'min_samples_split': [1.0, 2, 5], 
        'min_samples_leaf': [1, 3], 
        'max_features': ['auto', 'sqrt', 'log2']
        }
    
    # rf_params = {
    #     'n_estimators': [50, 100, 1000], 
    #     'max_depth': [5, 10, 20, 50, 100], 
    #     'max_features': ['auto', 'sqrt', 'log2']
    #     }
    
    
    rf_clf = GridSearchCV(rf, param_grid=rf_params,
                        scoring='recall',
                        n_jobs=-1,
                        cv=5)

    rf_clf.fit(X_train, y_train)

    rf_model = rf_clf.best_estimator_

    # save model
    pickle.dump(rf_model, open('models/random_forest_completion.p', 'wb')) 


    # evaluation
    predictions = rf_model.predict(X_test)
    roc_auc = roc_auc_score(y_test, predictions)
    probas = rf_model.predict_proba(X_test)[:, 1:]
    tprs, frps, thresh = roc_curve(y_test, probas)
    recall = recall_score(y_test, predictions)
    feat_imp = rf_model.feature_importances_

    print('Best Model: {}'.format(rf_model))
    print('Best Model parameters: {}'.format(rf_clf.best_params_))
    print('Best Model Log Recall: {}'.format(recall))
    print('Best Model Roc Auc: {}'.format(roc_auc))
    print('Feature Importances: {}'.format(feat_imp))

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
