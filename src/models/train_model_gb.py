
""" 
Functions -based solution
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, confusion_matrix, recall_score, roc_auc_score, roc_curve, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier


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

    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_train = y_train['module_not_completed']
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    y_test = y_test['module_not_completed']

    X_train.fillna(value = 0, inplace = True)
    X_test.fillna(value = 0, inplace = True)

    # estimator
    gb = GradientBoostingClassifier()
    
    # GridSearch parameters
    # gb_params = {
    #         'max_depth': [2, 3, 5],
    #         'learning_rate': [0.001, 0.01, 0.1],
    #         'n_estimators': [100, 500, 1000],
    #         'subsample': [0.5, 0.3, 0.1],
    #         'min_samples_split': [2, 5, 10],
    #         'min_samples_leaf': [1, 3, 5],
    #         'max_features': ['auto', 'sqrt', 'log2'],
    # }

    gb_params = {
            'max_depth': [3, 5, 10],
            'learning_rate': [0.001, 0.01, 0.1, 1],
            'n_estimators': [100, 1000],
            'subsample': [0.5, 0.3, 0.1],
            'max_features': ['auto', 'sqrt'],
    }

    gb_clf = GridSearchCV(gb, param_grid=gb_params,
                        scoring='roc_auc',
                        n_jobs=-1,
                        cv=5)

    gb_clf.fit(X_train, y_train)

    gb_model = gb_clf.best_estimator_

    # best parameters determined by grid search
    GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.01, loss='deviance', max_depth=10,
              max_features='auto', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=1000,
              n_iter_no_change=None, presort='auto', random_state=None,
              subsample=0.3, tol=0.0001, validation_fraction=0.1,
              verbose=0, warm_start=False)


    # save model
    # pickle.dump(gb_model, open('models/gradient_boost_completion_first_half.p', 'wb')) 


    # evaluation
    cross_val_score(gb_model, X_train, y_train, scoring = 'roc_auc', cv=5)
    cross_val_score(gb_model, X_train, y_train, scoring = 'recall', cv=5)
    cross_val_score(gb_model, X_train, y_train, scoring = 'precision', cv=5)
    cross_val_score(gb_model, X_train, y_train, scoring = 'accuracy', cv=5)


'''
    # final model evaluation (see jupyter notebook)
    predictions = gb_model.predict(X_test)
    roc_auc = roc_auc_score(y_test, predictions)
    probas = gb_model.predict_proba(X_test)[:, :1]
    tprs, fprs, thresh = roc_curve(y_test, probas)
    recall = recall_score(y_test, predictions)
    conf_mat = standard_confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    print_roc_curve(y_test, probas)
    print('Best Model: {}'.format(gb_model))
    print('\nRoc Auc: {}'.format(roc_auc))
    print('\nRecall Score: {}'.format(recall))
    print('\nClassification Report:\n {}'.format(class_report))
    print('\nConfusion Matrix:\n {}'.format(standard_confusion_matrix(y_test, predictions)))
'''
