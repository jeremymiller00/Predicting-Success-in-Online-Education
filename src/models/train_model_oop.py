
""" This solution makes heavy use of sklearn's Pipeline class.
    You can find documentation on using this class here:
    http://scikit-learn.org/stable/modules/pipeline.html
"""
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np
import pandas as pd

# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


class FillNaN(BaseEstimator, TransformerMixin):
    """
    Fill all NaN values with zero. NaN values are result of zero division in feature engineering. Columns affected: clicks_per_day, pct_days_vle_accesed max_clicks_one_day, first_date_vle_accessed, avg_score, avg_days_sub_early, estimated_final_score, days_early_first_assessment, score_first_assessment.
    """
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.fillna(value = 0)


class CustomScalar(BaseEstimator, TransformerMixin):
    '''
    Scale numeric data with sklearn StandardScalar
    '''
    def __init__(self):
        self.scalar = StandardScaler()

    def fit(self, X, y):
        self.numeric_cols = ['num_of_prev_attempts', 'studied_credits',
        'clicks_per_day', 'pct_days_vle_accessed','max_clicks_one_day',
        'first_date_vle_accessed', 'avg_score', 'avg_days_sub_early',   'days_early_first_assessment',
        'score_first_assessment']
        self.numeric = X[self.numeric_cols]
        self.categorical = X.drop(self.numeric_cols, axis = 1)
        # self.scalar.fit(self.numeric)
        return self

    def transform(self, X):
        X_scaled = pd.DataFrame(self.scalar.fit_transform(self.numeric))
        X_scaled.rename(columns = dict(zip(X_scaled.columns, self.numeric_cols)), inplace = True)
        return pd.concat([X_scaled, self.categorical], axis = 1)


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

########################################################################
if __name__ == '__main__':
    # fill na values
    # standard scalar
    # grid search cv
    # make base estimator a parameter

    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    # X_test = pd.read_csv('data/processed/X_test.csv')
    # y_test = pd.read_csv('data/processed/y_test.csv')
    X_train_mini = X_train.iloc[:100].drop('id_student', axis=1)
    y_train_mini = y_train['module_not_completed'].iloc[:100]


    p = Pipeline(steps = [
        ('fill_nan', FillNaN()),
        ('scaling', CustomScalar()),
        ('lr', LogisticRegression())
    ])

    # GridSearch
    params = {
            'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'lr__penalty': ['l2'],
            'lr__solver': ['newton-cg','lbfgs', 'liblinear'],
            'lr__max_iter': [25, 50, 100, 200],
            'lr__warm_start': ['False', 'True'],
    }

    clf = GridSearchCV(p, param_grid=params,
                        scoring='recall',
                        cv=5)

    clf.fit(X_train_mini, y_train_mini)

    model_dict = {}
    # models = [lr_clf, rf_clf, dt_clf, gb_clf, ada_clf, svc_clf]
    models = [lr_clf, rf_clf]
    for model in models:
        model_dict[model] = [model.best_score_]
    best_model, best_model_recall = max(model_dict.items(), key = lambda x: x[1])

    # test line
    # best_model = rf_clf

    print('Best Model: {}'.format(best_model))
    print('Best Model parameters: {}'.format(best_model.best_params_))
    print('Best Model Recall: {}'.format(best_model.best_score_))

    # save model
    pickle.dump(best_model, open('src/models/completion_classifier.p', 'wb')) 


    # test = pd.read_csv('data/test.csv')
    # test = test.sort_values(by='SalesID')

    # test_predictions = np.clip(clf.predict(test), y_min_cutoff, None)
    # test['SalePrice'] = test_predictions
    # outfile = 'data/solution_benchmark.csv'
    # test[['SalesID', 'SalePrice']].to_csv(outfile, index=False)