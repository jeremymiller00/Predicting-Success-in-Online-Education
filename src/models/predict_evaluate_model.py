'''
This script will use the Random Forest Classifier to make predictions and evaluate the model.
'''

import pickle
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import recall_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
from rfpimp import *
import collections as c
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='darkgrid')
plt.style.use('ggplot')
font = {'size'   : 30}
plt.rc('font', **font)
plt.ion()
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams["figure.figsize"] = (20.0, 10.0)
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 2000)
import warnings
warnings.filterwarnings('ignore')


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
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred, labels=[0,1])
    return np.array([[tp, fp], [fn, tn]])

def print_roc_curve(y_test, probabilities, model_name, roc_auc, recall):
    '''
    Calculates and prints a ROC curve given a set of test classes and probabilities from a trained classifier
    
    Parameters:
    ----------
    y_test: 1d array
    probabilities: 1d array of predeicted probabilites from X_test data
    model_name: name of model, for printing plot label
    recall: model recall score
    '''
    tprs, fprs, thresh = roc_curve(y_test, probabilities)
    plt.figure(figsize=(12,10))
    plt.plot(fprs, tprs, 
         label="AUC: {}\nRecall {}".format(round(roc_auc, 3), round(recall, 3)),
         color='blue', 
         lw=3)
    plt.plot([0,1],[0,1], 'k:')
    plt.legend(loc = 4, prop={'size': 30})
    plt.xlabel("FPR", fontsize=20)
    plt.ylabel("TPR", fontsize=20)
    plt.title("ROC Curve: {}".format(model_name), fontsize=40)
    
def print_confusion_matrix(conf_mat, model_name):
    '''
    Prints a formatted confusion matrix as a Seaborn heatmap with appropriate labels and titles.
    
    Parameters:
    ----------
    conf_mat: sklearn confusion matrix of classifier output
    model_name: name of model, for printing plot label
    
    '''
    plt.figure(figsize=(10,6))
    sns.heatmap(conf_mat, annot=conf_mat, cmap='coolwarm_r', fmt='d', annot_kws={"size": 40})
    plt.xlabel('Predicted Pass        Predicted Fail', fontsize=25)
    plt.ylabel('Actual Fail      Actual Pass', fontsize=25)
    plt.title('{} Confusion Matrix'.format(model_name), fontsize=40)
    plt.show()

def plot_target_hist(feature, h_range, bins=50, alpha=0.8):
    '''
    Plot a histogram of a feature with color indicating model prediction.
    
    feature: model feature as represented by a column in a dataframe
    h_range: range of values to print based on feature
    
    '''
    pos = np.where(y_test == True)
    neg = np.where(y_test == False)
    plt.figure(figsize=(12,8))
    plt.hist(X_test.loc[pos][feature], range=h_range, bins=bins, alpha=alpha, label='Fail')
    plt.hist(X_test.loc[neg][feature], range=h_range, bins=bins, alpha=alpha, label='Pass')
    plt.legend(prop={'size': 30})
    plt.title(feature, fontsize=30)

def plot_target_violin(df, feature):
    '''
    Plot a violin plt of a feature with color indicating model prediction.
    
    feature: model feature as represented by a column in a dataframe
    
    '''
    df1 = df.copy()
    df1['Predict Pass'] = y_test*-1+1
    plt.figure(figsize=(12,8))
    sns.violinplot(x = 'Predict Pass', y=feature, data=df1)
    plt.title(feature, fontsize=30)

def compare_hist(df1, df2, l1, l2):
    '''
    Prints overlayed histograms for comparison of distributions in two dataframes.
    
    Parameters:
    ----------
    df1: Pandas dataframe
    df2: Pandas dataframe with exact same columns as df1
    '''

    for col in df1.columns:
        plt.figure(figsize=(12,10))
        plt.hist(df1[col], bins=30, label=l1, alpha=0.6)
        plt.hist(df2[col], bins=30, label=l2, alpha=0.6)
        plt.legend()
        plt.title(col)
        plt.show()

if __name__ == "__main__":

    # load the model
    rf_model = pickle.load(open ('random_forest_completion_first_quarter.p', 'rb'))
    rf_model.get_params

    # load the data
    X_train = pd.read_csv('../data/processed/first_quarter/X_train.csv')
    y_train = pd.read_csv('../data/processed/first_quarter/y_train.csv')
    y_train = y_train['module_not_completed']
    X_test = pd.read_csv('../data/processed/first_quarter/X_test.csv')
    y_test = pd.read_csv('../data/processed/first_quarter/y_test.csv')
    y_test = y_test['module_not_completed']

    # fill
    X_train.fillna(value = 0, inplace = True)
    X_test.fillna(value = 0, inplace = True)

    t = 0.4 # threshold for predicting positive    
    predictions = (rf_model.predict_proba(X_test)[:, 1:] > t)
    roc_auc = roc_auc_score(y_test, predictions)
    probas = rf_model.predict_proba(X_test)[:, :1]
    tprs, fprs, thresh = roc_curve(y_test, probas)
    recall = recall_score(y_test, predictions, average='micro')
    conf_mat = confusion_matrix(y_test, predictions, labels=None) #     sklearn way
    class_report = classification_report(y_test, predictions)

    print_roc_curve(y_test, probas, 'Random Forest', roc_auc, recall)
    plt.savefig('../reports/figures/rf_roc.png')
    print_confusion_matrix(conf_mat, 'Classifier')
    plt.savefig('../reports/figures/rf_conf_mat.png')
    print('\nClassification Report:\n {}'.format(class_report))


    #Model Improvement Over Baseline: Predictions Based Solely on   demographics

    bl_cols = ['gender_M', 'gender_nan', 'region_East Midlands Region',     'region_Ireland', 'region_London Region', 'region_North Region',
           'region_North Western Region', 'region_Scotland','region_South   East Region', 'region_South Region', 'region_South West   Region', 
            'region_Wales', 'region_West Midlands Region',  'region_Yorkshire Region', 'region_nan',     'highest_education_HE Qualification',
           'highest_education_Lower Than A Level', 'highest_education_No    Formal quals', 'highest_education_Post Graduate Qualification',
           'highest_education_nan', 'imd_band_10-20', 'imd_band_20-30%',    'imd_band_30-40%', 'imd_band_40-50%', 'imd_band_50-60%',
           'imd_band_60-70%', 'imd_band_70-80%', 'imd_band_80-90%',     'imd_band_90-100%', 'imd_band_nan', 'age_band_35-55',   'age_band_55<=',
           'age_band_nan', 'disability_Y', 'disability_nan']

    bl_X_train = X_train[bl_cols]
    bl_X_test = X_test[bl_cols]
    bl_model = RandomForestClassifier()
    bl_model.fit(bl_X_train, y_train)

    t = 0.5 # threshold for predicting positive    
    bl_predictions = (bl_model.predict_proba(bl_X_test)[:, 1:] > t)
    bl_roc_auc = roc_auc_score(y_test, bl_predictions)
    bl_probas = bl_model.predict_proba(bl_X_test)[:, :1]
    bl_tprs, bl_fprs, bl_thresh = roc_curve(y_test, bl_probas)
    bl_recall = recall_score(y_test, bl_predictions, average='micro')
    bl_conf_mat = confusion_matrix(y_test, bl_predictions) # sklearn way
    bl_class_report = classification_report(y_test, bl_predictions)

    print_roc_curve(y_test, bl_probas, 'Baseline Thresholds', bl_roc_auc, bl_recall)
    plt.savefig('../reports/figures/bl_roc.png')
    print_confusion_matrix(bl_conf_mat, 'Classifier')
    plt.savefig('../reports/figures/bl_conf_mat.png')
    print('\nClassification Report:\n {}'.format(bl_class_report))


    # Permutation Feature Importance
    feat_imp = importances(rf_model, X_test, y_test)
    print("The ten most important features are:\n{}".format(feat_imp[:10])  )

    # Primary Features by Target
    plot_target_hist('avg_score', (0,100))
    plot_target_hist('sum_days_vle_accessed', (0,800))
    plot_target_hist('avg_days_sub_early', (-5,5))

    # Choosing a Threshold
    thresh_df = pd.DataFrame(data={'fprs': fprs, 'tprs': tprs,  'Thresholds': thresh}).loc[800:3487:200]
    print("A summary of how prediction rates are affected by threshold:\n   {}".format(thresh_df))

    t = 0.5 # threshold for predicting positive    
    predictions = (rf_model.predict_proba(X_test)[:, 1:] > t)
    class_report = classification_report(y_test, predictions)
    print_confusion_matrix(standard_confusion_matrix(y_test, predictions),   "T = 0.5")
    print('\nClassification Report at Threshold {}:\n {}'.format(t,     class_report))

    t = 0.4 # threshold for predicting positive    
    predictions = (rf_model.predict_proba(X_test)[:, 1:] > t)
    class_report = classification_report(y_test, predictions)
    print_confusion_matrix(standard_confusion_matrix(y_test, predictions),   "T = 0.4")
    print('\nClassification Report at Threshold {}:\n {}'.format(t,     class_report))

    t = 0.3 # threshold for predicting positive    
    predictions = (rf_model.predict_proba(X_test)[:, 1:] > t)
    class_report = classification_report(y_test, predictions)
    print_confusion_matrix(standard_confusion_matrix(y_test, predictions),   "T = 0.3")
    print('\nClassification Report at Threshold {}:\n {}'.format(t,     class_report))

    # Who is the model performing Poorly On?

    t = 0.4 # threshold for predicting positive    
    predictions = (rf_model.predict_proba(X_test)[:, 1:] > t)

    correct_prediction = predictions.ravel() == y_test

    X_test_with_pred = X_test

    X_test_with_pred['correct_prediction'] = correct_prediction

    X_test_corr_pred = X_test_with_pred[X_test_with_pred    ['correct_prediction'] == True]
    X_test_wrong_pred = X_test_with_pred[X_test_with_pred   ['correct_prediction'] == False]

    X_test_corr_pred.describe()

    X_test_wrong_pred.describe()

    compare_hist(X_test_wrong_pred, X_test_corr_pred, l1='Incorrect',   l2='Correct')

