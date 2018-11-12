
""" This solution makes heavy use of sklearn's Pipeline class.
    You can find documentation on using this class here:
    http://scikit-learn.org/stable/modules/pipeline.html
"""
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


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

class FilterColumns(BaseEstimator, TransformerMixin):
    """Drop duplicated index column.
    """
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.drop('Unnamed: 0', axis = 1)


class FillNaN(BaseEstimator, TransformerMixin):
    """
    Fill all NaN values with zero. NaN values are result of zero division in feature engineering. Columns affected: clicks_per_day, pct_days_vle_accesed max_clicks_one_day, first_date_vle_accessed, avg_score, avg_days_sub_early, estimated_final_score, days_early_first_assessment, score_first_assessment.
    """
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.fillna(value = 0, axis = 1, inplace = True)


class CustomScalar(TransformerMixin):
    '''
    Scale numeric data with sklearn StandardScalar
    '''
    def __init__(self):
        self.scalar = StandardScaler()

    def fit(self, X, y):
        self.scalar_cols = ['num_of_prev_attempts', 'studied_credits',
        'clicks_per_day', 'pct_days_vle_accessed', 'max_clicks_one_day',
        'first_date_vle_accessed', 'avg_score', 'avg_days_sub_early' 'days_early_first_assessment',
        'score_first_assessment']
        self.scalar.fit(X[self.scalar_cols], y)
        return self

    def transform(self, X):
        X_scaled = self.scalar.transform(X[self.scalar_cols])
        return X_scaled


def rmsle(y_hat, y, y_min=5000):
    """Calculate the root mean squared log error between y
    predictions and true ys.
    (hard-coding y_min for dumb reasons, sorry)
    """

    if y_min is not None:
        y_hat = np.clip(y_hat, y_min, None)
    log_diff = np.log(y_hat+1) - np.log(y+1)
    return np.sqrt(np.mean(log_diff**2))

###########################################################################
if __name__ == '__main__':
    # fill na values
    # standard scalar
    # grid search cv
    # define X and y: data should be already split
    # X is everything except finalresult, finalresultnum, modulenotcompleted, estimatedfinalscore
    # actually need three train / test / splits (completed, final result (numeric, ordered), estimated final score
    # separate models and grid searches for each
    X_train = pd.read_csv('data/processed/X_train')
    y_train = pd.read_csv('data/processed/X_train')
    X_test = pd.read_csv('data/processed/X_train')
    y_test = pd.read_csv('data/processed/X_train')

models = ['MLPClassifier', 'KNeighborsClassifier', 'SVC', 'GaussianProcessClassifier', 'RBF', 'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'GaussianNB', 'QuadraticDiscriminantAnalysis', 'LogisticRegression']


    p = Pipeline([
        ('fill_nan', FillNaN()),
        ('type_change', DataType()),
        ('replace_outliers', ReplaceOutliers()),
        ('compute_age', ComputeAge()),
        ('nearest_average', ComputeNearestMean()),
        ('columns', ColumnFilter()),
        ('lr', LogisticRegression())
    ])
    df = df.reset_index()

    # GridSearch
    params = {'nearest_average__window': [3, 5, 7]}

    # Turns our rmsle func into a scorer of the type required
    # by gridsearchcv.
    rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

    gscv = GridSearchCV(p, param_grid=params,
                        scoring=rmsle_scorer,
                        cv=cross_val,
                        n_jobs=-1)
    clf = gscv.fit(df.reset_index(), y)

    print('Best parameters: {}'.format(clf.best_params_))
    print('Best RMSLE: {}'.format(clf.best_score_))

    test = pd.read_csv('data/test.csv')
    test = test.sort_values(by='SalesID')

    test_predictions = np.clip(clf.predict(test), y_min_cutoff, None)
    test['SalePrice'] = test_predictions
    outfile = 'data/solution_benchmark.csv'
    test[['SalesID', 'SalePrice']].to_csv(outfile, index=False)