"""
Random Forest Regressor for Predicting Final Score
"""
import numpy as np
import pandas as pd
import pickle
import collections as c
# from rfpimp import *

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
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

def dropcol_importances(rf, X_train, y_train):
    '''
    Calculates the drop-column feature importances of a Random Forest model. Most precise, but computationally expensive because the model must be retrained. 
    Explanation here: https://explained.ai/rf-importance/index.html
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
    # make sure you have the correct data for time frame
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
    rf = RandomForestRegressor()

    rf_params = {
        'n_estimators': [50, 100, 1000], 
        'max_depth': [5, 10, 50, 100],
        'min_impurity_decrease': [0.01, 0.1, 0.5],
        'min_samples_leaf': [2, 3, 5 ,10],
        'min_samples_split': [2, 5, 10],
        'oob_score': ['True'],
        'max_features': ['auto', 'sqrt', 'log2']
        }

    rf_reg = RandomizedSearchCV(rf, 
                        param_distributions=rf_params,
                        n_iter = 20,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        cv=5)

    rf_reg.fit(X_train, y_train)

    rf_model = rf_reg.best_estimator_

    # best model as determined by grid search
    # rf_model = RandomForestRegressor(bootstrap=True, criterion='mse',                    max_depth=50,
    #        max_features='auto', max_leaf_nodes=None,
    #        min_impurity_decrease=0.1, min_impurity_split=None,
    #        min_samples_leaf=2, min_samples_split=10,
    #        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
    #        oob_score='True', random_state=None, verbose=0,
    #        warm_start=False)
    # rf_model.fit(X_train, y_train)

    # cross validation
    cv = cross_validate(rf_model,X_train,y_train,scoring='neg_mean_squared_error',cv=5,n_jobs=-1, verbose=1,return_train_score=1)
    print(cv)

    # evaluation
    mse_cv = (cross_val_score(rf_model, X_train, y_train, scoring = 'neg_mean_squared_error', cv=5))
    r2_cv = cross_val_score(rf_model, X_train, y_train, scoring = 'r2', cv=5)

    print('Best Model: {}'.format(rf_model))
    print('RMSE: {}'.format(np.sqrt(-1*mse_cv)))
    print('r2 Score: {}'.format(r2_cv))

    # save model
    pickle.dump(rf_model, open('models/random_forest_score_third_quarter.p', 'wb')) 

'''
    # final model evaluation (see jupyter notebook)
    predictions = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    evs = explained_variance_score(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print('Root Mean Squared Error: {}'.format(rmse))
    print('R-Squared: {}'.format(r2))
    print('Explained Variance Score: {}'.format(evs))

    # feature importances
    feat_imp = importances(rf_model, X_test, y_test)
    feat_imp.sort_values(by='Importance', ascending=False)[0:10]

    '''