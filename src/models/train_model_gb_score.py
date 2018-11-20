
""" 
Functions -based solution
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, confusion_matrix, recall_score, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV, cross_validate
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

def only_completed(X_train, y_train, X_test, y_test, y_train_not_comp, y_test_not_comp):
    '''
    Returns dataframes with only those students who completed the course for the purpose of regressing the final score.
    '''
    test_indices = []
    train_indices = []

    y_test_not_comp = y_test_not_comp[y_test_not_comp['module_not_completed'] == 1]
    for index, row in y_test_not_comp.iterrows():
        test_indices.append(index)

    y_train_not_comp = y_train_not_comp[y_train_not_comp['module_not_completed'] == 1]
    for index, row in y_train_not_comp.iterrows():
        train_indices.append(index)

    return X_train.drop(train_indices), y_train.drop(train_indices), X_test.drop(test_indices), y_test.drop(test_indices)

######################################################################

if __name__ == '__main__':
    # change path to get appropriate cutoff (third_quarter, third_quarter, third_quarter; CHANGE PATH IN WRITE OUT!)
    X_train = pd.read_csv('data/processed/third_quarter/X_train.csv')
    y_train = pd.read_csv('data/processed/third_quarter/y_train.csv')
    y_train_not_comp = y_train[['module_not_completed']]
    y_train = y_train['estimated_final_score']
    X_test = pd.read_csv('data/processed/third_quarter/X_test.csv')
    y_test = pd.read_csv('data/processed/third_quarter/y_test.csv')
    y_test_not_comp = y_test[['module_not_completed']]
    y_test = y_test['estimated_final_score']

    # fill
    X_train.fillna(value = 0, inplace = True)
    y_train.fillna(value = 0, inplace = True)
    X_test.fillna(value = 0, inplace = True)
    y_test.fillna(value = 0, inplace = True)

    # only students who completed the course
    X_train, y_train, X_test, y_test = only_completed(X_train, y_train, X_test, y_test, y_train_not_comp, y_test_not_comp)

    # estimator
    gb = GradientBoostingRegressor
    
    # GridSearch parameters

    gb_params = {
            'max_depth': [2, 3, 5],
            'learning_rate': [0.001, 0.01, 0.1, 1],
            'n_estimators': [100, 1000],
            'subsample': [0.5, 0.3, 0.1],
            'min_samples_leaf': [1, 5, 10, 50],
            'min_samples_split': [2, 10, 50, 100],
            'max_features': ['auto', 'sqrt'],
    }

    gb_clf = GridSearchCV(gb, 
                        param_grid=gb_params,,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        cv=5,
                        return_train_score=1)

    gb_clf.fit(X_train, y_train)

    gb_model = gb_clf.best_estimator_

    # best parameters determined by grid search
    # gb_model = GradientBoostingClassifier(criterion='friedman_mse', init=None,
    #           learning_rate=0.01, loss='deviance', max_depth=3,
    #           max_features='sqrt', max_leaf_nodes=None,
    #           min_impurity_decrease=0.0, min_impurity_split=None,
    #           min_samples_leaf=5, min_samples_split=50,
    #           min_weight_fraction_leaf=0.0, n_estimators=1000,
    #           presort='auto', random_state=None, subsample=0.5, verbose=0,
    #           warm_start=False)
    # gb_model.fit(X_train, y_train)

    # save model
    # pickle.dump(gb_model, open('models/gradient_boost_completion_third_quarter.p', 'wb')) 

    # cross validation
    cv = cross_validate(gb_model,X_train,y_train,scoring='neg_mean_squared_error',cv=5,n_jobs=-1, verbose=1,return_train_score=1)
    print(cv)

    # evaluation
    mse_cv = (cross_val_score(gb_model, X_train, y_train, scoring = 'neg_mean_squared_error', cv=5))
    r2_cv = cross_val_score(gb_model, X_train, y_train, scoring = 'r2', cv=5)

    print('Best Model: {}'.format(gb_model))
    print('RMSE: {}'.format(np.sqrt(-1*mse_cv)))
    print('r2 Score: {}'.format(r2_cv))


'''
    # final model evaluation (see jupyter notebook)
    predictions = gb_model.predict(X_test)
    roc_auc = roc_auc_score(y_test, predictions)
    probas = gb_model.predict_proba(X_test)[:, :1]
    tprs, fprs, thresh = roc_curve(y_test, probas)
    recall = recall_score(y_test, predictions)
    conf_mat = standard_confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    print_roc_curve(y_test, probas, 'Gradient Boosting')
    print('Best Model: {}'.format(gb_model))
    print('\nRoc Auc: {}'.format(roc_auc))
    print('\nRecall Score: {}'.format(recall))
    print('\nClassification Report:\n {}'.format(class_report))
    print('\nConfusion Matrix:\n {}'.format(standard_confusion_matrix(y_test, predictions)))

'''
