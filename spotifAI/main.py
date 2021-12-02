"""This script orchestrates the whole use case.
The steps that are globally taken are:
1. Retrieve the new tracks + features from the
new music friday playlist
2. Load in the trained model and predict for
 the new tracks what their relative ranking would be
3. Write the data including predictions to a cloud storage
bucket for later use (evaluating + retraining the model)
4. Sort the new tracks based on their predicted rank
 and publish the top 20 to Spotify"""

import pandas as pd
import requests
import json
from datetime import datetime
from flask import Flask
import waitress
from google.cloud import storage

# line below should be replaced with a new function once a trained model is available
from models.predict_model import predict_with_model


class SpotifAIapp:
    def __init__(self):

        self.model_name = "lgb_ranker.p"

        # urls below are hosted on cloud run
        self.get_new_music_friday_url = (
            "https://nmfscraper-fvfg5t6eda-ew.a.run.app/new_music_friday/"
        )
        self.publish_playlist_url = (
            "https://playlistpublisher-fvfg5t6eda-ew.a.run.app/publish_playlist/"
        )

        self.headers = {"content-type": "application/json"}

        # set up client that writes data to bucket
        self.client = storage.Client()
        self.bucket = self.client.get_bucket("spotifai_bucket")

    def run_app(self):

        # STEP 1: GET NEW DATA FROM NEW MUSIC FRIDAY PLAYLIST

        # get new music friday data and load it in a pandas Dataframe
        r = requests.post(self.get_new_music_friday_url, headers=self.headers)
        df = pd.DataFrame(json.loads(r.text))

        # STEP 2: PREDICT RELATIVE RANK IN CHARTS
        df = predict_with_model(
            df, model_bucket=self.bucket, model_name=self.model_name
        )

        # STEP 3: WRITE DATA TO CLOUD STORAGE BUCKET

        # save the dataframe as a csv in a folder with the date of today
        today = datetime.now()
        destination = f"new_music_friday_data/{today.strftime('%Y-%m-%d')}/nmf_data.csv"
        self.bucket.blob(destination).upload_from_string(
            df.to_csv(index=False), "text/csv"
        )

        # STEP 4: PUBLISH TOP 20 TO OUR SPOTIFY PLAYLIST
        # sort df on predicted hit position and put the 20
        # "best" track_ids in the request body
        request_body = {
            "track_ids": list(
                df.sort_values(by="score", ascending=False)["track_id"][:20].values
            )
        }

        # post request to refresh the spotify playlist
        r = requests.post(
            self.publish_playlist_url,
            data=json.dumps(request_body),
            headers=self.headers,
        )
        return r.text  # print returned response from publish_playlist end-point


if __name__ == "__main__":

    spotifai_app = SpotifAIapp()

    app = Flask(__name__)

    # Define API endpoints
    app.add_url_rule(
        "/run_app/", view_func=spotifai_app.run_app, methods=["POST"],
    )

    waitress.serve(app, port=8082)
