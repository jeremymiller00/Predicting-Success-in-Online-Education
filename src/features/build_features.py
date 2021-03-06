'''
Transformations, feature engineering and extraction

Inital dataframes imported in the if __name__ == '__main__' block are specified as keyword arguements for initial transformation (typically a join with a relevant table). Second level transformations and beyond are speficied with generic keyword arguements.

For changning cutoff date:
The factor needs to changed in three places:

join_vle: line 65
join_assessments: line 148
filter out: line 278
CHECK OUTPUT PATHS
'''


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# cast id to string
def to_string(X, cols):
    '''
    Casts specified columns of dataframe to type string.

    Parameters:
    ----------
    input {dataframe, list}: dataframe, list of columns to be cast as type string
    output {dataframe}: dataframe
    '''
    for col in cols:
        X[col] = X[col].astype('str')
    return X

# join student registration to student info
def join_reg_courses(main_df, reg_df, courses_df):
    '''
    Joins the student registrations table to the student info (master)table on three columns: code_module, code_presentation, id_student. Records without a value for 'date unregistration' are set to zero

    Parameters:
    --------
    input {dataframes}: studenInfo, studentRegistration
    output {dataframe}: joined dataframe
    '''
    df = pd.merge(main_df, reg_df, how='outer', on=['code_module', 'code_presentation', 'id_student'])
    df = pd.merge(df, courses_df, how='outer', on=['code_module', 'code_presentation'])
    return df

# join vle to student vle
def join_vle(st_vle_df, vle_df, courses_df):
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
    df =  pd.merge(st_vle_df, vle_df, how='outer', on = ['code_module', 'code_presentation', 'id_site'])
    df =  pd.merge(df, courses_df, how='outer', on = ['code_module', 'code_presentation'])
    # remove anything past halfway point
    df = df[df['date'] <= df['module_presentation_length'] * 0.75]
    return df

# create features from vle
def features_from_vle(df):
    '''
    Create model feaures from virtual learning environment data

    Parameters:
    ----------
    input {dataframe}: joined dataframe of all vle data
    output {dataframe}: dataframe for be joined to main df
    '''
    
    # caluculate sum clicks for each category
    total_clicks = df.groupby(by=['id_student', 'code_module', 'code_presentation', 'activity_type']).sum()[['sum_click']]
    cat_clicks = total_clicks.unstack(level=3, fill_value=0)
    cat_clicks.columns = cat_clicks.columns.get_level_values(1)
    new_cols = []
    for col in cat_clicks.columns:
        new_cols.append('sum_click_' + col)
    cat_clicks.columns = new_cols
    cat_clicks.reset_index(inplace=True)

    # calculate sum of days vle accesses
    days_accessed = df.groupby(by=['id_student', 'code_module', 'code_presentation']).count()[['date']]
    days_accessed.reset_index(inplace=True)
    days_accessed.rename({'date':'sum_days_vle_accessed'}, axis = 'columns',inplace=True)

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
    merged = pd.merge(cat_clicks, days_accessed, how='outer', on = ['code_module', 'code_presentation', 'id_student'])

    merged1 = pd.merge(merged, max_clicks, how='outer', on = ['code_module', 'code_presentation', 'id_student'])
    
    merged2 = pd.merge(merged1, first_accessed, how='outer', on = ['code_module', 'code_presentation', 'id_student'])

    return merged2


# join assessments to student assessments
def join_asssessments(st_asmt_df, asmt_df, courses_df):
    '''
    Joins the assessments table to the student assessment table on id_assessment; drop rows with null values (about 1.5%); relabel 'date' as 'due_date'.

    Parameters:
    ----------
    input {dataframes}: studentAssessment, assessments
    output {dataframe}: joined dataframe with new feature 'days_submited_early'.
    '''
    df = pd.merge(st_asmt_df, asmt_df, how='outer', on=['id_assessment']).dropna()
    df['id_student'] = df['id_student'].astype('int64')
    df =  pd.merge(df, courses_df, how='outer', on = ['code_module', 'code_presentation'])
    
    # add weighted score for each assessment
    df['weighted_score'] = df['score'] * df['weight'] / 100

    # calculate estimated final score; module DDD, presentations 2013J, 2013B, and 2014B are double modules, estimated final score should be cut in half
    f_df = df.groupby(by=['id_student', 'code_module', 'code_presentation']).sum()[['weighted_score']]
    f_df.reset_index(inplace=True)
    f_df.rename({'weighted_score':'estimated_final_score'}, axis = 'columns',inplace=True)

    #halving estimates for double modules
    indices = []
    double = f_df[(f_df['code_module'] == 'DDD') & ((f_df   ['code_presentation'] == '2013J') | (f_df['code_presentation'] ==  '2014B') | (f_df['code_presentation'] == '2013B'))]
    for index, row in double.iterrows():
        indices.append(index)
    double['estimated_final_score'] = double['estimated_final_score'].apply(lambda x: x/2)
    f_df.drop(indices, axis = 0, inplace = True)
    f_df = pd.concat([f_df, double])

    # remove anything past halfway point
    # must be done after final score calculated
    df = df[df['date'] <= df['module_presentation_length'] * 0.75]

    # this should be in the next function for best practice
    df['days_submitted_early'] = df['date'] - df['date_submitted']

    return pd.merge(df, f_df, how = 'outer', on = ['id_student', 'code_module', 'code_presentation'] )

# create features from assessments
def features_from_assessments(df):
    '''
    Returns per student/module/presentation averages of assessment score, days submitted early, and estimated final score

    Parameters:
    ----------
    input {dataframes}: joined assessment dataframe (from studentAssessment, assessment)
    output {dataframe}: df['avg_score', 'avg_days_submitted_early', 'est_final_score']
    '''

    # caluculate avg days submitted early, score
    av_df = df.groupby(by=['id_student', 'code_module', 'code_presentation']).mean()[['score', 'days_submitted_early']]
    av_df.reset_index(inplace=True)
    av_df.rename({'score': 'avg_score', 'days_submitted_early':'avg_days_sub_early'}, axis = 'columns',inplace=True)

    #test
    av_df.columns

    # first assessment days early and score
    early_first_assessment = df.groupby(by=['code_module', 'code_presentation', 'id_student']).max()[['days_submitted_early']]
    early_first_assessment.reset_index(inplace=True)
    early_first_assessment.rename({'days_submitted_early':'days_early_first_assessment'}, axis = 'columns',inplace=True)

    # test
    early_first_assessment.shape

    date_first_assessment = df.groupby(by=['code_module', 'code_presentation', 'id_student']).min()[['date']]
    date_first_assessment.reset_index(inplace=True)
    date_first_assessment.rename({'date':'date_first_assessment'}, axis = 'columns',inplace=True)

    # test
    date_first_assessment.shape

    temp_merged = pd.merge(df, date_first_assessment, how = 'outer', on = ['id_student', 'code_module', 'code_presentation'])

    score_first_assessment = temp_merged[temp_merged['date'] == temp_merged['date_first_assessment']][['code_module', 'code_presentation', 'id_student','score']]
    score_first_assessment.rename({'score':'score_first_assessment'}, axis = 'columns', inplace=True)

    # groupby df for estimated final score
    efs_df = df.groupby(by=['code_module', 'code_presentation', 'id_student']).last()[['estimated_final_score']]
    efs_df.reset_index(inplace=True)

    # merge dataframes
    merged = pd.merge(efs_df, av_df, how='outer', on = ['code_module', 'code_presentation', 'id_student'])

    merged1 = pd.merge(merged, early_first_assessment, how='outer', on = ['code_module', 'code_presentation', 'id_student'])

    final_assessment_df = pd.merge(merged1, score_first_assessment, how='outer', on = ['code_module', 'code_presentation', 'id_student'])

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
def encode_target(dataframe):
    '''
    Encodes target column 'final_result' into two categories from four.
    Retains original target column
    '''
    # create boolean: not completed y/n
    dataframe['module_not_completed'] = (dataframe['final_result'] == 'Fail') | (dataframe['final_result'] == 'Withdrawn')

    # create numeric column for final_result
    final_result_num = []
    for idx, row in dataframe.iterrows():
        if row['final_result'] == 'Withdrawn':
            final_result_num.append('0_Withdrawn')
        elif row['final_result'] == 'Fail':
            final_result_num.append('1_Fail')
        elif row['final_result'] == 'Pass':
            final_result_num.append('2_Pass')
        else:
            final_result_num.append('3_Distinction')

    dataframe['final_result_num'] = final_result_num

    return dataframe

####################################################################

if __name__ == "__main__":

    _cols_to_onehot = ['code_module', 'code_presentation', 'gender',    'region', 'highest_education', 'imd_band', 'age_band', 'disability', 'date_registration', ]

    # load in the dataframes
    main_df = pd.read_csv('data/raw/studentInfo.csv')
    courses_df = pd.read_csv('data/raw/courses.csv')
    reg_df = pd.read_csv('data/raw/studentRegistrations.csv')
    st_vle_df = pd.read_csv('data/raw/studentVle.csv')
    vle_df = pd.read_csv('data/raw/vle.csv')
    st_asmt_df = pd.read_csv('data/raw/studentAssessment.csv')
    asmt_df = pd.read_csv('data/raw/assessments.csv')

    # perfom transformations / feature engineering
    main_df = join_reg_courses(main_df, reg_df, courses_df)
    main_df = encode_target(main_df)
    joined_vle_df = join_vle(st_vle_df, vle_df, courses_df)
    features_vle = features_from_vle(joined_vle_df)
    joined_assessments = join_asssessments(st_asmt_df, asmt_df, courses_df)
    features_assessments = features_from_assessments(joined_assessments)

    # join dataframes to main_df    
    main_df = pd.merge(main_df, features_vle, how='outer', on=['code_module', 'code_presentation', 'id_student'])  
    
    main_df = pd.merge(main_df, features_assessments, how='outer', on = ['code_module', 'code_presentation', 'id_student'])

    # cast id_student to type string
    main_df = to_string(main_df, ['id_student'])

    # filter out students who dropped before the cutoff point
    main_df = main_df[(main_df['date_unregistration'] > main_df['module_presentation_length'] * 0.75) | (main_df['date_unregistration'].isnull())]

    # one-hot encode categorical variables
    main_df_final = one_hot(main_df, _cols_to_onehot)

    # split the data: three possible targets
    X = main_df_final.drop(['final_result', 'module_not_completed', 'final_result_num', 'estimated_final_score', 'date_unregistration', 'id_student'], axis = 1)

    y = main_df_final[['final_result', 'module_not_completed', 'final_result_num', 'estimated_final_score']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # write out to csv
    main_df_final.to_csv('data/processed/third_quarter/transformed_data_with_features.csv', index=False)
    X_train.to_csv('data/processed/third_quarter/X_train.csv', index=False)
    X_test.to_csv('data/processed/third_quarter/X_test.csv', index=False)
    y_train.to_csv('data/processed/third_quarter/y_train.csv', index=False)
    y_test.to_csv('data/processed/third_quarter/y_test.csv', index=False)

