"""
Linear Regression Model for Predicting Final Score
"""
import numpy as np
import pandas as pd
import pickle
import collections as c

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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

    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_train_not_comp = y_train[['module_not_completed']]
    y_train = y_train['estimated_final_score']
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    y_test_not_comp = y_test[['module_not_completed']]
    y_test = y_test['estimated_final_score']

    numeric_cols = ['num_of_prev_attempts', 'studied_credits', 'clicks_per_day', 'pct_days_vle_accessed','max_clicks_one_day','first_date_vle_accessed', 'avg_score', 'avg_days_sub_early','days_early_first_assessment', 'score_first_assessment']

    # fill
    X_train.fillna(value = 0, inplace = True)
    y_train.fillna(value = 0, inplace = True)
    X_test.fillna(value = 0, inplace = True)
    y_test.fillna(value = 0, inplace = True)

    # only students who completed the course
    X_train, y_train, X_test, y_test = only_completed(X_train, y_train, X_test, y_test, y_train_not_comp, y_test_not_comp)

    # # estimator
    # rf = RandomForestRegressor()

    # rf_params = {
    #     'n_estimators': [50, 100, 1000, 5000], 
    #     'max_depth': [5, 10, 20, 50, 100], 
    #     'max_features': ['auto']    
    # }
    
    # # rf_params = {
    # #     'n_estimators': [50, 100, 1000, 5000], 
    # #     'max_depth': [5, 10, 50, 100], 
    # #     'min_samples_split': [1.0, 2, 5], 
    # #     'min_samples_leaf': [1, 3], 
    # #     'max_features': ['auto', 'sqrt', 'log2']
    # #     }
    
    # rf_clf = GridSearchCV(rf, param_grid=rf_params,
    #                     scoring='neg_mean_squared_error',
    #                     n_jobs=-1,
    #                     cv=5)


    # rf_clf.fit(X_train, y_train)

    # rf_model = rf_clf.best_estimator_

    rf_model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=50, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,oob_score=False, random_state=None, verbose=0, warm_start=False)

    rf_model.fit(X_train, y_train)

    
    # evaluation
    predictions = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(rmse)
    evs = explained_variance_score(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print('RMSE: {}'.format(rmse))
    print('R-squared: {}'.format(r2))

    residuals = y_test - predictions

    plt.figure(figsize=(12,10))
    plt.scatter(x=residuals, y=y_test, alpha = 0.1, c='green')
    plt.show()

    plt.figure(figsize=(12,10))
    plt.hist(residuals, bins=100)
    plt.show()

    plt.figure(figsize=(12,10))
    plt.hist(y_train, bins=100)
    plt.show()

    np.std(y_test)

    feat_imp = rf_model.feature_importances_
    features = list(X_test.columns)
    coef_dict = c.OrderedDict((zip(feat_imp, features)))
    sorted(coef_dict.items(), reverse=True)

    # save model
    pickle.dump(rf_model, open('models/random_forest_score.p', 'wb'),-1)
