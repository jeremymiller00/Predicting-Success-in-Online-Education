
""" 
Functions -based solution
"""
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, confusion_matrix, recall_score, roc_auc_score, roc_curve, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

# from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


def fill_na_subset(df, columns):
    """
    Fill all NaN values with zero. NaN values are result of zero division in feature engineering. Columns affected: clicks_per_day, pct_days_vle_accesed max_clicks_one_day, first_date_vle_accessed, avg_score, avg_days_sub_early, estimated_final_score, days_early_first_assessment, score_first_assessment.
    """
    pass

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

    # local paths
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    # X_test = pd.read_csv('data/processed/X_test.csv')
    # y_test = pd.read_csv('data/processed/y_test.csv')


    X_train_mini = X_train.iloc[:10].drop('id_student', axis=1)
    y_train_mini = y_train['module_not_completed'].iloc[:10]

    numeric_cols = ['num_of_prev_attempts', 'studied_credits',
    'clicks_per_day', 'pct_days_vle_accessed','max_clicks_one_day',
    'first_date_vle_accessed', 'avg_score', 'avg_days_sub_early',  'days_early_first_assessment',
    'score_first_assessment']

    # fill and scale
    X_train_mini.fillna(value = 0, inplace = True)
    X_train_mini = scale_subset(X_train_mini, numeric_cols)

    # p = Pipeline(steps = [
    #     ('fill_nan', FillNaN()),
    #     ('scaling', CustomScalar()),

        # ('feature_proccessing', FeatureUnion(transformer_list = [
        #     ('categorical', FunctionTransformer(lambda data: data[categorical_cols])),
        #     ('numeric', Pipeline(steps = [
        #         ('select', FunctionTransformer(lambda data: data[numeric_cols])),
        #         ('scale', StandardScaler())
        #     ]))
        # ])),
    #     ('lr', LogisticRegression())
    # ])

    # estimators
    # lr = LogisticRegression()
    # rf = RandomForestClassifier()
    # dt = DecisionTreeClassifier()
    # gb = GradientBoostingClassifier()
    # ada = AdaBoostClassifier()
    svc = SVC()
    
    # GridSearch parameters
    # lr_params = {
    #         'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #         'penalty': ['l2'],
    #         'solver': ['newton-cg','lbfgs', 'liblinear'],
    #         'max_iter': [25, 50, 100, 200, 500, 1000],
    #         'warm_start': ['False', 'True'],
    # }

    # rf_params = {
    #         'n_estimators': [50, 100, 1000],
    #         'max_depth': [3, 5, 10],
    #         'min_samples_split': [2, 5, 10],
    #         'min_samples_leaf': [1, 3, 5],
    #         'max_features': ['auto', 'sqrt', 'log2'],
    #         'min_impurity_decrease': [0, 1, 5],
    # }
    
    # dt_params = {
    #         'max_depth': [3, 5, 10, 50],
    #         'min_samples_split': [2, 5, 10],
    #         'min_samples_leaf': [1, 3, 5],
    #         'max_features': ['auto', 'sqrt', 'log2'],
    #         'min_impurity_decrease': [0, 1, 5],
    # }
    
    # gb_params = {
    #         'max_depth': [2, 3, 5],
    #         'learning_rate': [0.001, 0.01, 0.1, 1],
    #         'n_estimators': [10, 100, 500, 1000, 5000],
    #         'subsample': [1, 0.5, 0.3, 0.1],
    #         'min_samples_split': [2, 5, 10],
    #         'min_samples_leaf': [1, 3, 5],
    #         'max_features': ['auto', 'sqrt', 'log2'],
    #         'min_impurity_decrease': [0, 1, 5],
    # }

    # gb_params = {
    #         'max_depth': [2, 3],
    #         'learning_rate': [0.1, 1],
    #         'n_estimators': [100, 1000],
    #         'subsample': [1, 0.5],
    #         'max_features': ['auto'],
    # }

    # ada_params = {
    #         'learning_rate': [0.001, 0.01, 0.1, 1],
    #         'n_estimators': [10, 100, 500, 1000, 5000],
    # }

    # svc_params = {
    #         'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #         'degree': [1, 3, 5],
    #         'gamma': [0.001, 0.01, 0.1],
    #         'probability': ['True'],
    # }

    svc_params = {
            'probability': 'True',
    }

    # lr_clf = GridSearchCV(lr, param_grid=lr_params,
    #                     scoring='recall',
    #                     n_jobs=-1,
    #                     cv=5)

    # rf_clf = GridSearchCV(rf, param_grid=rf_params,
    #                     scoring='recall',
    #                     n_jobs=-1,
    #                     cv=5)

    # dt_clf = GridSearchCV(dt, param_grid=dt_params,
    #                     scoring='recall',
    #                     n_jobs=-1,
    #                     cv=5)

    # gb_clf = GridSearchCV(gb, param_grid=gb_params,
    #                     scoring='recall',
    #                     n_jobs=-1,
    #                     cv=5)

    # ada_clf = GridSearchCV(ada, param_grid=ada_params,
    #                     scoring='recall',
    #                     n_jobs=-1,
    #                     cv=5)

    svc_clf = GridSearchCV(svc, param_grid=svc_params,
                        scoring='recall',
                        n_jobs=-1,
                        cv=5)

    # lr_clf.fit(X_train_mini, y_train_mini)
    # rf_clf.fit(X_train_mini, y_train_mini)
    # dt_clf.fit(X_train_mini, y_train_mini)
    # gb_clf.fit(X_train_mini, y_train_mini)
    # ada_clf.fit(X_train_mini, y_train_mini)
    svc_clf.fit(X_train_mini, y_train_mini)

    # print('Best LR parameters: {}'.format(lr_clf.best_params_))
    # print('Best LR Recall: {}'.format(lr_clf.best_score_))

    # model_dict = {}
    # models = [lr_clf, rf_clf, dt_clf, gb_clf, ada_clf, svc_clf]
    # # models = [lr_clf, rf_clf]
    # for model in models:
    #     model_dict[model] = [model.best_score_]
    # best_model, best_model_recall = max(model_dict.items(), key = lambda x: x[1])

    # test line
    best_model = sv_clf

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