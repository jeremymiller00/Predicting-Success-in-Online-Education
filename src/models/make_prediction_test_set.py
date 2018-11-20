'''
This scripts produces a set of 30 simulated observations which can be used to test the prediction scripts.
'''
import sys
import random
import pandas as pd

def shuffle_col_values(dataframe):
    '''
    Randomly shuffles the column values of the dataframe to produce simulated observations for testing purposes.
    
    Parameters:
    ----------
    input {dataframe}: Pandas dataframe object
    output {output}: Pandas dataframe object with column values, containing simulated observations
    '''
    for col in dataframe.columns:
        vals = list(sim_obs[col])
        random.shuffle(vals)
        dataframe[col] = vals
    return dataframe


################################################

if __name__ == '__main__':
    # read in the training data set
    sim_obs = pd.read_csv('data/processed/X_test.csv')

    # shuffle the column values randomly
    sim_obs = shuffle_col_values(sim_obs)

    # take a random sample of 100 observations
    sim_obs = sim_obs.sample(30).fillna(value=0)

    # write out the data
    sim_obs.to_csv('data/processed/prediction_test_set.csv')

    sys.exit()




