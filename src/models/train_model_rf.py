
""" 
Random Forest Classifier to predict course completion
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
# from rfpimp import *

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, confusion_matrix, recall_score, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
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

def print_roc_curve(y_test, probabilities, model_type):
    '''
    Calculates and prints a ROC curve given a set of test classes and probabilities from a trained classifier
    '''
    tprs, fprs, thresh = roc_curve(y_test, probabilities)
    plt.figure(figsize=(12,10))
    plt.plot(fprs, tprs, 
         label=model_type, 
         color='red')
    plt.plot([0,1],[0,1], 'k:')
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve AUC: {} Recall: {}".format(roc_auc, recall))
    plt.show()

######################################################################

if __name__ == '__main__':
    # change path to get appropriate cutoff (first_quarter, first_quarter, third_quarter; CHANGE PATH IN WRITE OUT!)
    X_train = pd.read_csv('data/processed/first_quarter/X_train.csv')
    y_train = pd.read_csv('data/processed/first_quarter/y_train.csv')
    y_train = y_train['module_not_completed']
    X_test = pd.read_csv('data/processed/first_quarter/X_test.csv')
    y_test = pd.read_csv('data/processed/first_quarter/y_test.csv')
    y_test = y_test['module_not_completed']

    X_train.fillna(value = 0, inplace = True)
    y_train.fillna(value = 0, inplace = True)
    X_test.fillna(value = 0, inplace = True)
    y_test.fillna(value = 0, inplace = True)

    # estimator
    # rf = RandomForestClassifier()
    
    # # GridSearch parameters
    # rf_params = {
    #     'n_estimators': [50, 100, 1000], 
    #     'max_depth': [5, 10, 50, 100], 
    #     'min_samples_split': [1.0, 10, 100], 
    #     'min_samples_leaf': [1, 10, 100], 
    #     'max_features': ['auto', 'sqrt', 'log2']
    #     }
    
    # rf_clf = GridSearchCV(rf, param_grid=rf_params,
    #                     scoring='roc_auc',
    #                     n_jobs=-1,
    #                     cv=5)

    # rf_clf.fit(X_train, y_train)
    # rf_model = rf_clf.best_estimator_

    # best model as determined by grid search
    rf_model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=30, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=10, min_samples_split=10, min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
    rf_model.fit(X_train, y_train)

    # cross validate
    cv = cross_validate(rf_model, X_train, y_train, scoring = 'roc_auc', cv=5, return_train_score=1)
    print(cv)

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
    pickle.dump(rf_model, open('models/random_forest_completion_first_quarter.p', 'wb')) 

'''
    # final model evaluation (see jupyter notebook)
    predictions = rf_model.predict(X_test)
    roc_auc = roc_auc_score(y_test, predictions)
    probas = rf_model.predict_proba(X_test)[:, :1]
    tprs, fprs, thresh = roc_curve(y_test, probas)
    recall = recall_score(y_test, predictions)
    conf_mat = standard_confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    print_roc_curve(y_test, probas, 'Random Forest')
    print('Best Model: {}'.format(rf_model))
    print('\nRoc Auc: {}'.format(roc_auc))
    print('\nRecall Score: {}'.format(recall))
    print('\nClassification Report:\n {}'.format(class_report))
    print('\nConfusion Matrix:\n {}'.format(standard_confusion_matrix(y_test, predictions)))

    # feature importances
    feat_imp = importances(rf_model, X_test, y_test)
    feat_imp.sort_values(by='Importance', ascending=False)[0:10]
    pd.DataFrame(data={'fprs': fprs, 'tprs': tprs, 'Thresholds': thresh}).loc[300:1000:25]
    '''