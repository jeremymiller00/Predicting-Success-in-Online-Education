# # Exploratory Data Analysis of Engineered Features

# Observations:
# * Moderate to weak visual correlations between engineered features and targets
# * Score of first assessment looks to have a sigificant collelation with estimated final score, final outcome
# * Individual distributions show a lot of 0 values, unsurprising otherwise

import numpy as np 
import pandas as pd
import seaborn as sns
sns.reset_defaults
sns.set_style(style='darkgrid')
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
font = {'size'   : 16}
plt.rc('font', **font)
plt.ion()
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["patch.force_edgecolor"] = True
plt.rcParams['figure.figsize'] = (20.0, 10.0)
pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 2000)

df = pd.read_csv('../data/processed/transformed_data_with_features.csv')
df.fillna(value=0, inplace=True)

# ## Engineered Feature Analysis

# ### Bivariate Plots Against Targets: final_result_num, estimated_final_score

sns.jointplot(x='clicks_per_day', y='final_result_num', data=df)

sns.jointplot(x='clicks_per_day', y='estimated_final_score', data=df)

sns.jointplot(x='pct_days_vle_accessed', y='final_result_num', data=df)

sns.jointplot(x='pct_days_vle_accessed', y='estimated_final_score', data=df)

sns.jointplot(x='studied_credits', y='final_result_num', data=df)

sns.jointplot(x='studied_credits', y='estimated_final_score', data=df)

sns.jointplot(x='max_clicks_one_day', y='final_result_num', data=df)

sns.jointplot(x='max_clicks_one_day', y='estimated_final_score', data=df)

sns.jointplot(x='first_date_vle_accessed', y='final_result_num', data=df)

sns.jointplot(x='first_date_vle_accessed', y='estimated_final_score', data=df)

sns.jointplot(x='avg_days_sub_early', y='final_result_num', data=df)

sns.jointplot(x='avg_days_sub_early', y='estimated_final_score', data=df)

sns.jointplot(x='days_early_first_assessment', y='final_result_num', data=df)

sns.jointplot(x='days_early_first_assessment', y='estimated_final_score', data=df)

sns.jointplot(x='score_first_assessment', y='final_result_num', data=df)

sns.jointplot(x='score_first_assessment', y='estimated_final_score', data=df)

sns.jointplot(x='avg_score', y='final_result_num', data=df)

sns.jointplot(x='avg_score', y='estimated_final_score', data=df)

# ### Univariate Plots of Engineered Features

f = list(df.columns[[6,7,8,9,10,11,12,13,14]])

for feat in f:
    plt.figure()
    sns.distplot(df[feat])

