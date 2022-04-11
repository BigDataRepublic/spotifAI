"""Predict hit potential for tracks using a trained model.

This script loads in the trained model and predicts a ranking
for the tracks.
"""
import pickle
from tempfile import TemporaryFile

import numpy as np
import pandas as pd
from google.cloud import storage


def predict_random(df: pd.DataFrame) -> pd.DataFrame:
    """Randomly predicts a ranking for a dataframe with tracks.

    Args:
        df: a dataframe with tracks to rank

    Returns:
        object: the dataframe with a column containing a randomly assigned rank
    """
    df = df.assign(rank=np.random.randint(1, 202, df.shape[0]))
    return df


def predict_with_model(
    df: pd.DataFrame, model_bucket: storage.bucket.Bucket, model_name: str
) -> pd.DataFrame:
    """Predicts a ranking for a dataframe with tracks with a trained model.

    The model is trained on historical data of global top 200 hits.

    Args:
        df: a dataframe with tracks to rank
        model_bucket: the cloud bucket containing the trained model
        model_name: the name of the model to use for predictions

    Returns:
        object: the dataframe with a column containing the predicted rank
    """
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
    df = df.assign(score=loaded_model.predict(df[predictor_variables]))

    return df
