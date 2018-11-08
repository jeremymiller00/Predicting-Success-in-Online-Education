'''
Transformations, feature engineering and extraction
'''

import pandas as pd
import numpy as np

# convert to string? id, module, presentation

# make dummmies: result, gender, region, highest_ed, imd_band, age_band, diability

# fill in null values in imd_band? use other features to predict?

cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']

def drop_nulls(dataframe):
    '''
    Drops rows with null values from dataframe
    '''
    return dataframe.dropna(axis = 0)


def one_hot(dataframe, columns):
    '''
    Concatenates dummy variable (one-hot encoded) columns onto original dataframe for specified columns. Original columns are dropped.

    Parameters:
    ----------
    input {dataframe}: 
    output {list}: list of columns to be one-hot encoded and dropped
    '''

    dumms = pd.get_dummies(dataframe[columns], dummy_na=True, drop_first=True)
    full_df = pd.concat([dataframe, dumms], axis = 1)
    return full_df.drop(cols, axis = 1)

def _encode_target(dataframe):
    '''
    Encodes target column 'final_result' into two categories from four.
    Retains original target column
    '''
    dataframe['module_completed'] = (dataframe['final_result'] == 'Pass') | (dataframe['final_result'] == 'Distinction')
    return dataframe   