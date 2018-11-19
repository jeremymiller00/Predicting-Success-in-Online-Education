"""
Linear Regression Model for Predicting Final Score
"""
import numpy as np
import pandas as pd
import collections as c
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
# import scipy.stats as scs
import matplotlib.pyplot as plt

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        self.residuals_ = self.results_.outlier_test()['student_resid']
    def fit_regularized(self, X, y, a, l1):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit_regularized(alpha=a, L1_wt=l1)
    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)
    def summary(self):
        return self.results_.summary()

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

def plot_residuals(residuals, target, features, data):
    '''
    Creates scatterplots of residuals against target and specified features
    Parameters:
    ----------
    input {all array-like}: calculated residuals, target values from test data, list of features to plot against
    output {plots}: some plots
    '''
    plt.figure(figsize=(12,10))
    plt.scatter(x=y_test, y=residuals, alpha = 0.1, c='red')
    plt.xlabel("Residuals")
    plt.ylabel("Target")
    plt.title("Residuals vs. Target")
    plt.show()

    for f in features:
        plt.figure(figsize=(12,10))
        plt.scatter(x=data[f], y=residuals, alpha=0.1, c='blue')
        plt.xlabel("Residuals")
        plt.ylabel(f)
        plt.title("Residuals vs {}".format(f))
        plt.show()

### test lines DELETE!
cd Galvanize/dsi-capstone
ls
%reset
whos
######################################################################

if __name__ == '__main__':

    X_train = pd.read_csv('data/processed/first_half/X_train.csv')
    y_train = pd.read_csv('data/processed/first_half/y_train.csv')
    y_train_not_comp = y_train[['module_not_completed']]
    y_train = y_train['estimated_final_score']
    X_test = pd.read_csv('data/processed/first_half/X_test.csv')
    y_test = pd.read_csv('data/processed/first_half/y_test.csv')
    y_test_not_comp = y_test[['module_not_completed']]
    y_test = y_test['estimated_final_score']

    numeric_cols = ['num_of_prev_attempts', 'studied_credits', 'module_presentation_length', 'sum_click_dataplus', 'sum_click_dualpane', 'sum_click_externalquiz', 'sum_click_forumng','sum_click_glossary', 'sum_click_homepage', 'sum_click_htmlactivity', 'sum_click_oucollaborate', 'sum_click_oucontent', 'sum_click_ouelluminate', 'sum_click_ouwiki', 'sum_click_page', 'sum_click_questionnaire', 'sum_click_quiz', 'sum_click_repeatactivity', 'sum_click_resource', 'sum_click_sharedsubpage', 'sum_click_subpage', 'sum_click_url', 'sum_days_vle_accessed', 'max_clicks_one_day', 'first_date_vle_accessed', 'avg_score', 'avg_days_sub_early', 'days_early_first_assessment', 'score_first_assessment']

    # fill and scale
    X_train.fillna(value = 0, inplace = True)
    y_train.fillna(value = 0, inplace = True)
    X_train = scale_subset(X_train, numeric_cols)
    # X_train = sm.add_constant(X_train)
    X_test.fillna(value = 0, inplace = True)
    y_test.fillna(value = 0, inplace = True)
    X_test = scale_subset(X_test, numeric_cols)
    # X_test = sm.add_constant(X_test)

    # only students who completed the course
    X_train, y_train, X_test, y_test = only_completed(X_train, y_train, X_test, y_test, y_train_not_comp, y_test_not_comp)

    # resolve multicolinearity
    '''
    [(34.82408402588052, 'code_presentation_2014J'),
    (27.902556097941535, 'module_presentation_length'),
    (17.71032093366156, 'sum_days_vle_accessed'),
    (12.29325003243407, 'avg_score'),
    (12.129102421725705, 'code_module_BBB'),
    (11.26607390128323, 'score_first_assessment'),
    (10.74221505000013, 'code_module_DDD'),
    (10.185010067236496, 'code_module_FFF')'''

    high_vif = ['sum_click_repeatactivity', 'days_early_first_assessment', 'sum_days_vle_accessed', 'module_presentation_length', 'score_first_assessment']
    X_train.drop(high_vif, axis = 1, inplace = True)
    X_test.drop(high_vif, axis = 1, inplace = True)
    
    # estimator
    lin_reg_model = SMWrapper(sm.OLS)
    lin_reg_model.fit(X_train, y_train)
    lin_reg_model.summary()

    # evaluation
    rmse_cv = np.sqrt((cross_val_score(lin_reg_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)))
    r2_cv = cross_val_score(lin_reg_model, X_train, y_train, scoring='r2', cv=5)

    print('CV Root Mean Squared Error: {}'.format(np.sqrt(mse_cv)))
    print('CV R-Squared: {}'.format(r2_cv))

    # check for homoscedasticity
    f_statistic, p_value, _ = sm.stats.diagnostic.het_goldfeldquandt(y_train, X_train, idx=1, alternative='two-sided')
    print(p_value)


    # save model
    pickle.dump(lin_reg_model_model, open('models/linear_regression_first_half.p', 'wb'))

    # assessing variance inflation
    vif = []
    for v in range(len(X_train.columns)):
        vif.append(variance_inflation_factor(X_train.values, v))
    features = list(X_test.columns)
    vif_dict = c.OrderedDict((zip(vif, features)))
    sorted(vif_dict.items(), reverse=True)[:10]

    # feature correlation
    cor = X_train.corr().abs()
    s = cor.unstack()
    so = s.sort_values(kind="quicksort", ascending=False)
    so[58:150:2]

    # plot residuals
    stud_resid = lin_reg_model.residuals_
    plot_residuals(stud_resid, y_train, X_train.columns, X_train)

    # QQ-plot
    ax = sm.graphics.qqplot(stud_resid, line='45')

'''
    # final model evaluation (see jupyter notebook)
    predictions = lin_reg_model.predict(X_test)
    residuals = predictions - y_test
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    evs = explained_variance_score(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print('Root Mean Squared Error: {}'.format(rmse))
    print('R-Squared: {}'.format(r2))
    print('Explained Variance Score: {}'.format(evs))

    plot_residuals(residuals, y_test, X_test.columns, X_test)
    ax = sm.graphics.qqplot(residuals, line='45')

    plt.hist(residuals, bins=100)
    plt.show()
'''