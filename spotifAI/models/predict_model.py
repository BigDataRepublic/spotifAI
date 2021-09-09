"""This script loads in the trained model
and predicts the maximum future position in the
global spotify top 200 chart for new tracks"""
import numpy as np


def predict_random(df):
    """Takes in a dataframe and adds a column 'max_position'
    filled with random values between 1 & 201"""
    df = df.assign(max_position=np.random.randint(1, 202, df.shape[0]))
    return df
