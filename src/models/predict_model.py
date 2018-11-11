

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


class DataType(BaseEstimator, TransformerMixin):
    """
    Cast the data types of the id and data source columns to strings
    from numerics.
    """
    col_types = {'str': ['id_student']}

    def fit(self, X, y):
        return self

    def transform(self, X):
        for col_type, column in self.col_types.items():
            X[column] = X[column].astype(col_type)
        return X


class FillNaN(BaseEstimator, TransformerMixin):
    """
    Fill all NaN values with zero. NaN values are result of zero division in feature engineering. Columns affected: clicks_per_day, pct_days_vle_accesed max_clicks_one_day, first_date_vle_accessed, avg_score, avg_days_sub_early, estimated_final_score, days_early_first_assessment, score_first_assessment.
    """
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.fillna(value = 0, axis = 1, inplace = True)


def rmsle(y_hat, y, y_min=5000):
    """Calculate the root mean squared log error between y
    predictions and true ys.
    (hard-coding y_min for dumb reasons, sorry)
    """

    if y_min is not None:
        y_hat = np.clip(y_hat, y_min, None)
    log_diff = np.log(y_hat+1) - np.log(y+1)
    return np.sqrt(np.mean(log_diff**2))


if __name__ == '__main__':

    # define X and y: data should be already split
    # actually need three train / test / splits (completed, final result (numeric, ordered), estimated final score
    # separate models and grid searches for each
    X_train = pd.read_csv('data/processed/X_train')
    y_train = pd.read_csv('data/processed/X_train')
    X_test = pd.read_csv('data/processed/X_train')
    y_test = pd.read_csv('data/processed/X_train')

    p = Pipeline([
        ('data_type', DataType()),
        ('fill_nan', FillNaN()),
        ('lm', LinearRegression())
    ])
   
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