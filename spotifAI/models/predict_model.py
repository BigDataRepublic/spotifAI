"""This script loads in the trained model
and predicts the maximum future position in the
global spotify top 200 chart for new tracks"""
import numpy as np
import pickle
from tempfile import TemporaryFile

def predict_random(df):
    """Takes in a dataframe and adds a column 'max_position'
    filled with random values between 1 & 201"""
    df = df.assign(max_position=np.random.randint(1, 202, df.shape[0]))
    return df

def predict_with_model(df, model_bucket, model_name):
    """Takes in a dataframe and adds a column 'max_position',
    predicted by a trained classifier that is loaded in from the
    storage bucket with the provided modelname argument"""

    full_modelpath = f"models/{model_name}"

    blob = model_bucket.blob(full_modelpath)

    with TemporaryFile() as temp_file:
        # download blob into temp file
        blob.download_to_file(temp_file)
        temp_file.seek(0)

        # load pickled model
        loaded_model = pickle.load(temp_file)

    # predict with same variables that model was trained on
    predictor_variables = loaded_model.feature_names

    # predict and assign prediction as new column
    df = df.assign(
        max_position=[round(pred) for pred in loaded_model.predict(df[predictor_variables])])

    return df


