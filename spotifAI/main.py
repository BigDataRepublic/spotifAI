"""This script orchestrates the whole Spotifai app.

The steps that are globally taken are:
1. Retrieve the new tracks + features from the
new music friday playlist
2. Load in the trained model and predict for
the new tracks what their relative ranking would be
3. Write the data including predictions to a cloud storage
bucket for later use (evaluating + retraining the model)
4. Sort the new tracks based on their predicted rank
and publish the top 20 to Spotify
"""

import json
import logging
from datetime import datetime

import pandas as pd
import requests  # type: ignore
import waitress  # type: ignore
from flask import Flask
from google.cloud import storage
from models.predict_model import predict_with_model
from requests.adapters import HTTPAdapter, Retry  # type: ignore


class SpotifAIapp:
    def __init__(self) -> None:

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

    def run_app(self) -> str:
        """Function to run the app that determines the content of the playlist to publish.

        Returns:
            object: a text with the result of the request to publish the playlist
        """
        # STEP 1: GET NEW DATA FROM NEW MUSIC FRIDAY PLAYLIST
        s = requests.Session()

        retries = Retry(
            total=3, backoff_factor=2, status_forcelist=[500], raise_on_status=True
        )

        s.mount("https://", HTTPAdapter(max_retries=retries))

        r = s.post(self.get_new_music_friday_url, headers=self.headers)

        df = pd.DataFrame(json.loads(r.text))

        # STEP 2: PREDICT RELATIVE RANK IN CHARTS
        df = predict_with_model(
            df, model_bucket=self.bucket, model_name=self.model_name
        )

        # STEP 3: WRITE DATA TO CLOUD STORAGE BUCKET
        today = datetime.now()
        destination = f"new_music_friday_data/{today.strftime('%Y-%m-%d')}/nmf_data.csv"
        self.bucket.blob(destination).upload_from_string(
            df.to_csv(index=False), "text/csv"
        )

        # STEP 4: PUBLISH TOP 20 TO OUR SPOTIFY PLAYLIST
        request_body = {
            "track_ids": list(
                df.sort_values(by="score", ascending=False)["track_id"][:20].values
            )
        }

        r = requests.post(
            self.publish_playlist_url,
            data=json.dumps(request_body),
            headers=self.headers,
        )
        return r.text


if __name__ == "__main__":

    FORMAT = "%(asctime)s|%(levelname)s|%(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    spotifai_app = SpotifAIapp()
    app = Flask(__name__)

    # Define API endpoints
    app.add_url_rule("/run_app/", view_func=spotifai_app.run_app, methods=["POST"])

    waitress.serve(app, port=8083)
