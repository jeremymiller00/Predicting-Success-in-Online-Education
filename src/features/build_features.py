'''
Transformations, feature engineering and extraction

Inital dataframes imported in the if __name__ == '__main__' block are specified as keyword arguements for initial transformation (typically a join with a relevant table). Second level transformations and beyond are speficied with generic keyword arguements.
'''

import pandas as pd
import numpy as np

# join student registration to student info
def _join_reg(main_df, reg_df):
    '''
    Joins the student registrations table to the student info (master)table on three columns: code_module, code_presentation, id_student. Records without a value for 'date unregistration' are set to zero

    Parameters:
    --------
    input {dataframes}: studenInfo, studentRegistration
    output {dataframe}: joined dataframe
    '''
    return pd.merge(main_df, reg_df, how='outer', on=['code_module', 'code_presentation', 'id_student']).fillna(value = 0)

# join vle to student vle
def _join_vle(st_vle_df, vle_df):
    '''
    Joins vle table to studentVle table. Drops columns which are mostly null values.

    Parameters:
    --------
    input {dataframes}: vle, studentVle dataframes
    output {dataframe}: joined dataframe
    '''
    # drop columns with mostly null values
    # vle_df.drop(['week_from', 'week_to'], axis = 1, inplace = True)

    # merge together
    return pd.merge(st_vle_df, vle_df, how='outer', on = ['code_module', 'code_presentation', 'id_site'])

# create features from vle
def _features_from_vle(df):
    '''
    Create model feaures from virtual learning environment data

    Parameters:
    ----------
    input {dataframe}: joined dataframe of all vle data
    output {dataframe}: dataframe for be joined to main df
    '''
    
    # caluculate clicks/day (total clicks / course length)
    total_clicks = df.groupby(by=['id_student', 'code_module', 'code_presentation']).sum()[['sum_click']]
    total_clicks.reset_index(inplace=True)
    total_clicks.rename({'sum_click':'total_clicks_in_course'}, axis = 'columns',inplace=True)
    total_clicks = pd.merge(total_clicks, courses_df, how="outer", on = ['code_module', 'code_presentation'])
    total_clicks['clicks_per_day'] = total_clicks['total_clicks_in_course'] / total_clicks['module_presentation_length']
    total_clicks.drop(['total_clicks_in_course', 'module_presentation_length'], axis = 1, inplace = True)

    # calculate percentage of days vle accesses (sum_days_accessed / course length)***
    days_accessed = df.groupby(by=['id_student', 'code_module', 'code_presentation']).count()[['date']]
    days_accessed.reset_index(inplace=True)
    days_accessed.rename({'date':'sum_days_vle_accessed'}, axis = 'columns',inplace=True)
    days_accessed = pd.merge(days_accessed, courses_df, how="outer", on = ['code_module', 'code_presentation'])
    days_accessed['pct_days_vle_accessed'] = days_accessed['sum_days_vle_accessed'] / days_accessed['module_presentation_length']
    days_accessed.drop(['sum_days_vle_accessed', 'module_presentation_length'], axis = 1, inplace = True)

    # calculate max clicks in one day
    max_clicks = df.groupby(by=['id_student',
    'code_module', 'code_presentation']).max()[['sum_click']]
    max_clicks.reset_index(inplace=True)
    max_clicks.rename({'sum_click':'max_clicks_one_day'}, axis = 'columns',inplace=True)

    # first date vle accessed
    first_accessed = df.groupby(by=['id_student', 'code_module', 'code_presentation']).min()[['date']]
    first_accessed.reset_index(inplace=True)
    first_accessed.rename({'date':'first_date_vle_accessed'}, axis = 'columns',inplace=True)

    # merge and rename columns
    merged = pd.merge(total_clicks, days_accessed, how='outer', on = ['code_module', 'code_presentation', 'id_student'])

    merged1 = pd.merge(merged, max_clicks, how='outer', on = ['code_module', 'code_presentation', 'id_student'])
    
    merged2 = pd.merge(merged1, first_accessed, how='outer', on = ['code_module', 'code_presentation', 'id_student'])

    return merged2

# join assessments to student assessments
def _join_asssessments(st_asmt_df, asmt_df):
    '''
    Joins the assessments table to the student assessment table on id_assessment; drop rows with null values (about 1.5%); relabel 'date' as 'due_date'.

    Parameters:
    ----------
    input {dataframes}: studentAssessment, assessments
    output {dataframe}: joined dataframe with new feature 'days_submited_early'.
    '''
    merged = pd.merge(st_asmt_df, asmt_df, how='outer', on=['id_assessment']).dropna()
    merged['id_student'] = merged['id_student'].astype('int64')
    # this should be in the next function for best practice
    merged['days_submitted_early'] = merged['date'] - merged['date_submitted']
    return merged

# create features from assessments
def _features_from_assessments(df):
    '''
    Returns per student/module/presentation averages of assessment score, days submitted early, and estimated final score

    Parameters:
    ----------
    input {dataframes}: joined assessment dataframe (from studentAssessment, assessment)
    output {dataframe}: df['avg_score', 'avg_days_submitted_early', 'est_final_score']
    '''

    # add weighted score for each assessment
    df['weighted_score'] = df['score'] * df['weight'] / 100

    # caluculate mean scores, mean days submitted early
    av_df = df.groupby(by=['id_student', 'code_module', 'code_presentation']).mean()[['score', 'days_submitted_early']]
    av_df.reset_index(inplace=True)
    av_df.rename({'score':'avg_score', 'days_submitted_early':'avg_days_sub_early'}, axis = 'columns',inplace=True)

    # calculate estimated final score
    f_df = df.groupby(by=['id_student', 'code_module', 'code_presentation']).sum()[['weighted_score']]
    f_df.reset_index(inplace=True)
    f_df.rename({'weighted_score':'estimated_final_score'}, axis = 'columns',inplace=True)

    # first assessment date and score
    early_first_assessment = df.groupby(by=['code_module', 'code_presentation', 'id_student']).max()[['days_submitted_early']]
    early_first_assessment.reset_index(inplace=True)
    early_first_assessment.rename({'days_submitted_early':'days_early_first_assessment'}, axis = 'columns',inplace=True)

    date_first_assessment = df.groupby(by=['code_module', 'code_presentation', 'id_student']).min()[['date']]
    date_first_assessment.reset_index(inplace=True)
    date_first_assessment.rename({'date':'date_first_assessment'}, axis = 'columns',inplace=True)

    temp_merged = pd.merge(df, date_first_assessment, how = 'outer', on = ['id_student', 'code_module', 'code_presentation'])

    score_first_assessment = temp_merged[temp_merged['date'] == temp_merged['date_first_assessment']][['code_module', 'code_presentation', 'id_student','score']]
    score_first_assessment.rename({'score':'score_first_assessment'}, axis = 'columns', inplace=True)

    # merge dataframes
    merged = pd.merge(av_df, f_df, how='outer', on = ['code_module', 'code_presentation', 'id_student'])

    merged1 = pd.merge(merged, early_first_assessment, how='outer', on = ['code_module', 'code_presentation', 'id_student'])

    merged2 = pd.merge(merged1, date_first_assessment, how='outer', on = ['code_module', 'code_presentation', 'id_student'])

    final_assessment_df = pd.merge(merged2, score_first_assessment, how='outer', on = ['code_module', 'code_presentation', 'id_student'])

    return final_assessment_df
    
# make dummmies
def one_hot(dataframe, columns):
    '''
    Concatenates dummy variable (one-hot encoded) columns onto original dataframe for specified columns. Original columns are dropped.

    Parameters:
    ----------
    input {dataframe, list}: original dataframe, list of columns to be one-hot encoded and dropped
    output {dataframe}: resulting modified dataframe
    '''
    dumms = pd.get_dummies(dataframe[columns], dummy_na=True, drop_first=True)
    full_df = pd.concat([dataframe, dumms], axis = 1)
    return full_df.drop(columns, axis = 1)

# encode target: pass/fail
# three potential targets: pass/fail, type of result, esi final score
def _encode_target(dataframe):
    '''
    Encodes target column 'final_result' into two categories from four.
    Retains original target column
    '''
    dataframe['module_not_completed'] = (dataframe['final_result'] == 'Fail') | (dataframe['final_result'] == 'Withdrawn')
    return dataframe

####################################################################
if __name__ == "__main__":

    _cols_to_onehot = ['code_module', 'code_presentation', 'gender',    'region', 'highest_education', 'imd_band', 'age_band', 'disability', 'date_registration', ]

    # load in the dataframes
    main_df = pd.read_csv('data/raw/studentInfo.csv')
    courses_df = pd.read_csv('data/raw/courses.csv')
    reg_df = pd.read_csv('data/raw/studentRegistrations.csv')
    st_asmt_df = pd.read_csv('data/raw/studentAssessment.csv')
    asmt_df = pd.read_csv('data/raw/assessments.csv')
    st_vle_df = pd.read_csv('data/raw/studentVle.csv')
    vle_df = pd.read_csv('data/raw/vle.csv')

    # perfom transformations / feature engineering
    main_df = _join_reg(main_df, reg_df)
    main_df = _encode_target(main_df)
    joined_vle_df = _join_vle(st_vle_df, vle_df)
    features_vle = _features_from_vle(joined_vle_df)
    joined_assessments = _join_asssessments(st_asmt_df, asmt_df)
    features_assessments = _features_from_assessments(joined_assessments)

    # join dataframes to main_df    
    main_df = pd.merge(main_df, features_vle, how='outer', on=['code_module', 'code_presentation', 'id_student'])  
    
    main_df = pd.merge(main_df, features_assessments, how='outer', on = ['code_module', 'code_presentation', 'id_student'])

    # one-hot encode categorical variables
    main_df_final = one_hot(main_df, _cols_to_onehot)

    # write out to csv
    main_df_final.to_csv('data/processed/transformed_data_with_features.csv')