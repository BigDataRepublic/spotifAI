"""This script orchestrates the whole use case.
The steps that are globally taken are:
1. Retrieve the new tracks + features from the
new music friday playlist
2. Load in the trained model and predict for
 the new tracks what their future maximum position will be
3. Sort the new tracks based on their predicted future max.
 position and publish the top 20 to Spotify"""

import pandas as pd
import requests
import json

# line below should be replaced with a new function once a trained model is available
from models.predict_model import predict_random


# url below is hosted on cloud run
get_new_music_friday_url = "https://nmfscraper-fvfg5t6eda-ew.a.run.app/new_music_friday/"
headers = {"content-type": "application/json"}
# get new music friday data and load it in a pandas Dataframe
r = requests.post(get_new_music_friday_url, headers=headers)
df = pd.DataFrame(json.loads(r.text))

# Code below should be replaced once a trained model is available #
# add column 'max_position', for now filled with random predictions
df = predict_random(df)

# TODO: write data from new music friday to bucket with predictions

# sort df on predicted hit position and put the 20 "best" track_ids in the request body
request_body = {
    "track_ids": list(df.sort_values(by="max_position")["track_id"][:20].values)
}

# the url below is hosted by running the Dockerfile in the deployment folder
# will need to be replaced with a different url once application is running on cloud
publish_playlist_url = "https://playlistpublisher-fvfg5t6eda-ew.a.run.app/publish_playlist/"

# post request to refresh the spotify playlist
r = requests.post(publish_playlist_url, data=json.dumps(request_body), headers=headers)
print(r.text)  # print returned response (including link to playlist)
