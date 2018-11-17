
""" 
Functions -based solution
"""
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, confusion_matrix, recall_score, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# try pipeline with fill and scale, one at a time

class FillNaN(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.fillna(value = 0, inplace=False)


class ScaleSubset(BaseEstimator, TransformerMixin):
    '''
        Use sklearn StandardScalar to scale only numeric columns.
    
        Parameters:
        ----------
        input {dataframe, list}: dataframe containing mixed feature variable    types, list of names of numeric feature columns
        output: {dataframe}: dataframe with numeric features scaled and     categorical features unchanged
    
        '''
    def __init__(self):
        self.numeric_columns = ['num_of_prev_attempts', 'studied_credits', 'clicks_per_day', 'pct_days_vle_accessed', 'max_clicks_one_day', 'first_date_vle_accessed', 'avg_score', 'avg_days_sub_early', 'days_early_first_assessment', 'score_first_assessment']
        self.scalar = StandardScaler()

    def fit(self, X, y=None):
        self.numeric = X[self.numeric_columns]
        self.categorical = X.drop(self.numeric_columns, axis = 1)
        return self
    
    def transform(self, X):
        self.scalar.fit(self.numeric)
        num_scaled = pd.DataFrame(self.scalar.transform(self.numeric))
        num_scaled.rename(columns = dict(zip(num_scaled.columns,    self.numeric_columns)), inplace = True)
        return pd.concat([num_scaled, self.categorical], axis = 1)
    

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

    # read in the data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_train = y_train[['module_not_completed']]
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    y_test = y_test[['module_not_completed']]

    # small test data sets
    X_train = X_train[:100]
    y_train = y_train[:100]
    X_test = X_test[:100]

    f = FillNaN()
    s = ScaleSubset()
    lr = LogisticRegression()


    p = Pipeline([
        ('fill_nulls', f),
        ('scale', s),
        ('lr', lr)
    ])
    
    # GridSearch parameters
    # lr_params = {
    #         'C': [0.1, 1, 10, 100],
    #         'penalty': ['l2'],
    #         'solver': ['lbfgs', 'liblinear'],
    #         'max_iter': [100, 200, 500],
    #         'warm_start': ['True'],
    # }

    lr_params = {
            'lr__C': [1, 10],
    }

    lr_clf = GridSearchCV(p, param_grid=lr_params,
                        scoring='neg_log_loss',
                        cv=5)

    # p.fit(X_train, y_train)

    lr_clf.fit(X_train, y_train)

    # log_reg_model = lr_clf.best_estimator_

    # best model as determined by grid search
    # model = LogisticRegression(C=1, class_weight=None, dual=False,    fit_intercept=True, intercept_scaling=1, max_iter=200,            multi_class='warn', n_jobs=None, penalty='l2', random_state=None, solver='newton-cg', tol=0.0001, verbose=0, warm_start='False')

    # save model
    # pickle.dump(log_reg_model, open('~/Desktop/logistic_regression_completion_pipeline.p', 'wb'))

    # # evaluation
    # predictions = log_reg_model.predict(X_test)
    # roc_auc = roc_auc_score(y_test, predictions)
    # probas = log_reg_model.predict_proba(X_test)[:, 1:]
    # tprs, fprs, thresh = roc_curve(y_test, probas)
    # recall = recall_score(y_test, predictions)
    # conf_mat = standard_confusion_matrix(y_test, predictions)
    # class_report = classification_report(y_test, predictions)

    # print_roc_curve(y_test, probas)
    # print('Best Model: {}'.format(log_reg_model))
    # print('Best Model parameters: {}'.format(lr_clf.best_params_))
    # print('Best Model Log Loss: {}'.format(lr_clf.best_score_))
    # print('Roc Auc: {}'.format(roc_auc))
    # print('Recall Score: {}'.format(recall))
    # print('Confusion Matrix: {}'.format(conf_mat))
    # print('Classification Report: {}'.format(class_report))

    # # Feature Importances
    # abs_coef = list(np.abs(log_reg_model.coef_.ravel()))
    # features = list(X_test.columns)
    # coef_dict = c.OrderedDict((zip(abs_coef, features)))
    # ordered_feature_importances = sorted(coef_dict.items(), reverse=True)
    # print('The top ten features affecting completion are: {}'.format( ordered_feature_importances[:10])
    