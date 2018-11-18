
""" 
Functions -based solution
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
# from rfpimp import *

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, confusion_matrix, recall_score, roc_auc_score, roc_curve, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score
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

    X_train.fillna(value = 0, inplace = True)
    X_test.fillna(value = 0, inplace = True)

    # estimator
    rf = RandomForestClassifier()
    
    # GridSearch parameters
    # rf_params = {
    #     'n_estimators': [50, 100, 1000], 
    #     'max_depth': [5, 10, 50], 
    #     'min_samples_split': [1.0, 10, 100], 
    #     'min_samples_leaf': [1, 10, 100], 
    #     'max_features': ['auto', 'sqrt', 'log2']
    #     }
    
    # rf_params = {
    #     'n_estimators': [50, 100, 1000], 
    #     'max_depth': [5, 10, 20, 50, 100, 500], 
    #     'max_features': ['auto', 'sqrt', 'log2']
    #     }
    
    # rf_clf = GridSearchCV(rf, param_grid=rf_params,
    #                     scoring='recall',
    #                     n_jobs=-1,
    #                     cv=5)

    # rf_clf.fit(X_train, y_train)
    # rf_model = rf_clf.best_estimator_

    # best model as determined by grid search
    rf_model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=100, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None, oob_score=False, random_state=None, verbose=0, warm_start=False)
    rf_model.fit(X_train, y_train)

    # evaluation
    roc_auc_cv = (cross_val_score(rf_model, X_train, y_train, scoring = 'roc_auc', cv=5))
    recall_cv = cross_val_score(rf_model, X_train, y_train, scoring = 'recall', cv=5)
    precision_cv = cross_val_score(rf_model, X_train, y_train, scoring = 'precision', cv=5)
    accuracy_cv = cross_val_score(rf_model, X_train, y_train, scoring = 'accuracy', cv=5)
    f1_cv = cross_val_score(rf_model, X_train, y_train, scoring = 'f1_micro', cv=5)

    print('Best Model: {}'.format(rf_model))
    # print('Best Model parameters: {}'.format(rf_model.best_params_))
    print('Roc Auc: {}'.format(roc_auc_cv))
    print('Recall Score: {}'.format(recall_cv))
    print('Precision Score: {}'.format(precision_cv))
    print('Accuracy Score: {}'.format(accuracy_cv))
    print('F1 Micro: {}'.format(f1_cv))

    # save model
    pickle.dump(rf_model, open('models/random_forest_completion_first_half.p', 'wb')) 

'''
    # final model evaluation (see jupyter notebook)
    predictions = rf_model.predict(X_test)
    roc_auc = roc_auc_score(y_test, predictions)
    probas = rf_model.predict_proba(X_test)[:, :1]
    tprs, fprs, thresh = roc_curve(y_test, probas)
    recall = recall_score(y_test, predictions)
    conf_mat = standard_confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    print_roc_curve(y_test, probas)
    print('Best Model: {}'.format(rf_model))
    print('\nRoc Auc: {}'.format(roc_auc))
    print('\nRecall Score: {}'.format(recall))
    print('\nClassification Report:\n {}'.format(class_report))
    print('\nConfusion Matrix:\n {}'.format(standard_confusion_matrix(y_test, predictions)))

    # feature importances
    importances(rf_model, X_test, y_test)


'''