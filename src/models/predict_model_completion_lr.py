'''
For when you get a new datapoint. Data must first be transformed by build_features.py
'''

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

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


######################################################################

if __name__ == '__main__':

    data = pd.read_csv('data/processed/prediction_test_set.csv', index_col=0)

    numeric_cols = ['num_of_prev_attempts', 'studied_credits', 'module_presentation_length', 'sum_click_dataplus', 'sum_click_dualpane', 'sum_click_externalquiz', 'sum_click_forumng','sum_click_glossary', 'sum_click_homepage', 'sum_click_htmlactivity', 'sum_click_oucollaborate', 'sum_click_oucontent', 'sum_click_ouelluminate', 'sum_click_ouwiki', 'sum_click_page', 'sum_click_questionnaire', 'sum_click_quiz', 'sum_click_repeatactivity', 'sum_click_resource', 'sum_click_sharedsubpage', 'sum_click_subpage', 'sum_click_url', 'sum_days_vle_accessed', 'max_clicks_one_day', 'first_date_vle_accessed', 'avg_score', 'avg_days_sub_early', 'days_early_first_assessment', 'score_first_assessment']

    # fill and scale
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    data_filled = data.fillna(value=0)
    data_filled_scaled = scale_subset(data_filled, numeric_cols)

    high_vif = ['module_presentation_length', 'sum_days_vle_accessed','score_first_assessment', 'days_early_first_assessment', 'sum_click_folder']
    data_filled_scaled.drop(high_vif, axis = 1, inplace = True)

    # load the model
    model = pickle.load(open('models/logistic_regression_completion_first_half.p', 'rb'))

    pred = model.predict(data_filled_scaled)
    data['Will the Student Complete?'] = ~pred
    print('\n')
    print(data[['Will the Student Complete?', 'num_of_prev_attempts', 'studied_credits', 'avg_score', 'avg_days_sub_early']])

'''
    data[['Will the Student Complete?', 'num_of_prev_attempts', 'studied_credits', 'sum_days_vle_accessed', 'avg_score', 'avg_days_sub_early', 'code_module_BBB', 'code_module_CCC', 'code_module_DDD', 'code_module_EEE', 'code_module_FFF', 'code_module_GGG', 'code_module_nan', 'code_presentation_nan', 'gender_M', 'gender_nan', 'region_East Midlands Region', 'region_Ireland',
    'region_London Region', 'region_North Region', 'region_North Western Region', 'region_Scotland',
    'region_South East Region', 'region_South Region', 'region_South West Region', 'region_Wales',
    'region_West Midlands Region', 'region_Yorkshire Region', 'region_nan', 'highest_education_HE Qualification', 'highest_education_Lower Than A Level', 'highest_education_No Formal quals',
    'highest_education_Post Graduate Qualification', 'highest_education_nan', 'imd_band_10-20', 'imd_band_20-30%', 'imd_band_30-40%', 'imd_band_40-50%', 'imd_band_50-60%', 'imd_band_60-70%', 'imd_band_70-80%', 'imd_band_80-90%', 'imd_band_90-100%', 'imd_band_nan', 'age_band_35-55', 'age_band_55<=', 'age_band_nan', 'disability_Y', 'disability_nan']]
    '''
