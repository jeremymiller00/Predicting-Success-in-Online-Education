'''
Transformations, feature engineering and extraction
'''

import pandas as pd
import numpy as np



cols = ['code_module', 'code_presentation', 'gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']

# drop null values (about 3.5% of rows)
def drop_nulls(dataframe):
    '''
    Drops rows with null values from dataframe
    '''
    return dataframe.dropna(axis = 0)

# join student registration to student info
def _join_reg(df1, df2):
    '''
    Joins the student registrations table to the student info (master)table on three columns: code_module, code_presentation, id_student.
    '''
    return pd.merge(s_df, r_df, how='outer', on=['code_module', 'code_presentation', 'id_student'])

# join vle to student vle
# join vle to student info


# join assessments to student assessments
# join joint assessments to studentinfo
# calculate final grade for courses completed


# make dummmies: module pres, result, gender, region, highest_ed, imd_band, age_band, diability
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

# encode target: pass/fail

def _encode_target(dataframe):
    '''
    Encodes target column 'final_result' into two categories from four.
    Retains original target column
    '''
    dataframe['module_not_completed'] = (dataframe['final_result'] == 'Fail') | (dataframe['final_result'] == 'Withdrawn')
    return dataframe



if __name__ == "__main__":
    # import the dataframes

    # join assessments
    pd.merge(std_asmt_df, asmt_df, how='outer', on='id_assessment')