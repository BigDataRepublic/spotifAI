"""This script orchestrates the whole use case.
The steps that are globally taken are:
1. Retrieve the new tracks + features from the
new music friday playlist
2. Load in the trained model and predict for
 the new tracks what their future maximum position will be
3. Write the data including predictions to a cloud storage
bucket for later use (evaluating + retraining the model)
4. Sort the new tracks based on their predicted future max.
 position and publish the top 20 to Spotify"""

import pandas as pd
import requests
import json
from datetime import datetime
from flask import Flask
import waitress
from google.cloud import storage

# line below should be replaced with a new function once a trained model is available
from models.predict_model import predict_random


def run_app():

    # STEP 1: GET NEW DATA FROM NEW MUSIC FRIDAY PLAYLIST
    # url below is hosted on cloud run
    get_new_music_friday_url = (
        "https://nmfscraper-fvfg5t6eda-ew.a.run.app/new_music_friday/"
    )
    headers = {"content-type": "application/json"}
    # get new music friday data and load it in a pandas Dataframe
    r = requests.post(get_new_music_friday_url, headers=headers)
    df = pd.DataFrame(json.loads(r.text))

    # STEP 2: PREDICT FUTURE MAXIMUM POSITION IN CHARTS
    # Code below should be replaced once a trained model is available #
    # add column 'max_position', for now filled with random predictions
    df = predict_random(df)

    # STEP 3: WRITE DATA TO CLOUD STORAGE BUCKET
    # set up client that writes data to bucket
    client = storage.Client()
    bucket = client.get_bucket("spotifai_bucket")

    # save the dataframe as a parquet file in a folder with the date of today
    today = datetime.now()
    destination = f"new_music_friday_data/{today.strftime('%Y-%m-%d')}/nmf_data.csv"
    bucket.blob(destination).upload_from_string(df.to_csv(index=False), "text/csv")

    # STEP 4: PUBLISH TOP 20 TO OUR SPOTIFY PLAYLIST
    # sort df on predicted hit position and put the 20
    # "best" track_ids in the request body
    request_body = {
        "track_ids": list(df.sort_values(by="max_position")["track_id"][:20].values)
    }

    publish_playlist_url = (
        "https://playlistpublisher-fvfg5t6eda-ew.a.run.app/publish_playlist/"
    )

    # post request to refresh the spotify playlist
    r = requests.post(
        publish_playlist_url, data=json.dumps(request_body), headers=headers
    )
    return r.text  # print returned response from publish_playlist end-point


if __name__ == "__main__":

    app = Flask(__name__)

    # Define API endpoints
    app.add_url_rule(
        "/run_app/", view_func=run_app, methods=["POST"],
    )

    waitress.serve(app, port=8082)
