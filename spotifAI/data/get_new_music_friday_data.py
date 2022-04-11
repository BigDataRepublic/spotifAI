"""Script that scrapes the global Spotify playlist 'new music friday'.

100 newly released tracks are fetched from the playlist and enriched
with additional features obtained via the Spotify API.
"""

import logging
import time
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import spotipy
import waitress  # type: ignore
from flask import Flask
from google.cloud import secretmanager
from spotipy.oauth2 import SpotifyClientCredentials
from urllib3.exceptions import MaxRetryError, ReadTimeoutError


class Scraper:
    """Class to scrape tracks and features with the Spotify API."""

    def __init__(self) -> None:
        cid = self.access_secret_version(
            "projects/420207002838/secrets/SPOTIFY_CLIENT_ID/versions/1"
        )
        secret = self.access_secret_version(
            "projects/420207002838/secrets/SPOTIFY_CLIENT_SECRET/versions/1"
        )
        client_credentials_manager = SpotifyClientCredentials(
            client_id=cid, client_secret=secret
        )
        self.spotify = spotipy.Spotify(
            client_credentials_manager=client_credentials_manager
        )

    @staticmethod
    def access_secret_version(secret_version_id: str) -> str:
        """Return the value of a secret's version.

        Args:
            secret_version_id: the id of the secret version in the secret manager

        Returns:
            object: the secret decoded in utf-8
        """
        client = secretmanager.SecretManagerServiceClient()

        response = client.access_secret_version(name=secret_version_id)

        return response.payload.data.decode("UTF-8")

    def get_tracks_from_playlist(self, playlist_url: str) -> pd.DataFrame:
        """Queries data for a specific playlist url.

        Args:
            playlist_url: the url of the playlist to query

        Returns:
            object: a dataframe with one track per row
        """
        track_dicts = self.spotify.playlist(playlist_url)["tracks"]["items"]

        if track_dicts is None:
            raise TypeError(
                "Failed to get the playlist information from Spotify API. "
                "Aborting operation."
            )

        rows = []

        for track_dict in track_dicts:
            date_added = track_dict["added_at"].split("T")[0]

            # all other interesting variables are inside the 'track' dict
            track_dict = track_dict["track"]

            track_id = track_dict["id"]

            # track_dict["artists"] can contain multiple elements
            # if multiple artists are on a track.
            # they are concatenated into one string with separator " ft. "
            track_artist = " ft. ".join(
                [artist_dict["name"] for artist_dict in track_dict["artists"]]
            )
            track_name = track_dict["name"]

            popularity = track_dict["popularity"]
            spotify_url = track_dict["external_urls"]["spotify"]

            row_dict = {
                "track_id": track_id,
                "artist": track_artist,
                "name": track_name,
                "track_popularity": popularity,
                "date_added": date_added,
                "spotify_url": spotify_url,
            }

            rows.append(row_dict)

        playlist_df = pd.DataFrame.from_records(rows)
        playlist_df["date_of_scrape"] = datetime.now().strftime("%Y-%m-%d")

        return playlist_df

    def get_audio_features(self, track_id: str) -> dict:
        """Returns the audio features for a given track_id.

        Args:
            track_id: the id of the track

        Returns:
            object: a dictionary with audio features
        """
        return self.spotify.audio_features(track_id)

    def get_other_track_info(self, track_id: str) -> Tuple[str, bool, List[str]]:
        """Gets other track information for a given track_id.

        More specifically, the release date, a boolean indicating whether explicit
        language is used in a track and a list with artist ids.

        Args:
            track_id: the id of the track

        Returns:
            object: a tuple with the release date, explicit boolean and artist ids
        """
        track_object = self.spotify.track(track_id)
        release_date = self.get_release_date(track_object)
        explicit_boolean = self.get_explicit_boolean(track_object)
        artist_ids = self.get_artist_ids(track_object)
        return release_date, explicit_boolean, artist_ids

    @staticmethod
    def get_release_date(track_object: dict) -> str:
        """Returns the release date for a given track.

        Args:
            track_object: dictionary containing information about a track

        Returns:
            object: the release date of a track
        """
        return track_object["album"]["release_date"]

    @staticmethod
    def get_explicit_boolean(track_object: dict) -> bool:
        """Returns whether the track contains explicit language.

        Args:
            track_object: dictionary containing information about a track

        Returns:
            object: a boolean indicating explicit language in a track
        """
        return track_object["explicit"]

    @staticmethod
    def get_artist_ids(track_object: dict) -> List[str]:
        """Returns the ids of the artists that appear on a track.

        Args:
            track_object: dictionary containing information about a track

        Returns:
            object: a list with one or more artist ids
        """
        track_artists_object = track_object["artists"]
        return [artist["id"] for artist in track_artists_object]

    def get_artist_features(self, artist_ids: List[str]) -> Tuple[int, int]:
        """Gets features on artist-level.

        Args:
            artist_ids: list of artist ids that can contain on or more artists

        Returns:
            object: a tuple with the number of followers and the artist popularity
        """
        artists_objects = [self.spotify.artist(uid) for uid in artist_ids]
        followers = self.get_followers(artists_objects)
        artist_popularity = self.get_artist_popularity(artists_objects)
        return followers, artist_popularity

    @staticmethod
    def get_followers(artists_objects: List[dict]) -> int:
        """Gets the number of followers of an artist.

        If multiple artists participate in a track,
        the sum of the number of followers of every artist is returned.

        Args:
            artists_objects: a list with dicts containing information about the artist.

        Returns:
            object: the sum of followers of the artist(s)
        """
        artists_followers = [artist["followers"]["total"] for artist in artists_objects]
        return sum(artists_followers)

    @staticmethod
    def get_artist_popularity(artists_objects: List[dict]) -> int:
        """Gets the popularity of an artist.

        If multiple artists participate in a track, the average of the popularity of
        every artist is returned, rounded to the nearest integer.

        Args:
            artists_objects: a list with dicts containing information about the artist.

        Returns:
            object: the artist's popularity
        """
        return round(
            sum([artist["popularity"] for artist in artists_objects])
            / len(artists_objects)
        )

    def get_playlist_with_features(self, playlist_id: str) -> pd.DataFrame:
        """Get all tracks in a playlist with the features used for modelling.

        Args:
            playlist_id: the id of the playlist to collect features for

        Returns:
            object: a dataframe with per row a track and corresponding features
        """
        nmf = self.get_tracks_from_playlist(playlist_id)

        # collect audio features for the nmf tracks
        audio_features = (
            pd.concat(
                [
                    pd.DataFrame(self.get_audio_features(track_id=uid))
                    for uid in nmf.track_id.values
                ]
            )
            .rename(columns={"id": "track_id"})
            .drop(columns=["type", "uri", "track_href", "analysis_url"])
        )

        # merge nmf tracks with audio features
        df = pd.merge(nmf, audio_features, on="track_id")

        # add other track info (release_date, explicit, artist_ids)
        release_date_dict = {}
        explicit_dict = {}
        artist_ids_dict = {}

        for track_id in df.track_id.values:
            release_date, explicit, artist_ids = self.get_other_track_info(track_id)
            release_date_dict[track_id] = release_date
            explicit_dict[track_id] = explicit
            artist_ids_dict[track_id] = artist_ids

        # map values to columns
        df = (
            df.assign(
                release_date=lambda x: x["track_id"].apply(
                    lambda uid: release_date_dict.get(uid)
                )
            )
            .assign(
                explicit=lambda x: x["track_id"].apply(
                    lambda uid: explicit_dict.get(uid)
                )
            )
            .assign(
                artist_ids=lambda x: x["track_id"].apply(
                    lambda uid: artist_ids_dict.get(uid)
                )
            )
        )

        # add artist-level features
        artist_followers_dict = {}
        artist_popularity_dict = {}
        for track_id in df.track_id.values:
            followers, artist_popularity = self.get_artist_features(
                df.loc[df.track_id == track_id, "artist_ids"].values[0]
            )
            artist_followers_dict[track_id] = followers
            artist_popularity_dict[track_id] = artist_popularity

        # map values to columns
        df = df.assign(
            followers=lambda x: x["track_id"].apply(
                lambda uid: artist_followers_dict.get(uid)
            )
        ).assign(
            artist_popularity=lambda x: x["track_id"].apply(
                lambda uid: artist_popularity_dict.get(uid)
            )
        )

        return df

    def get_new_music_friday(self) -> str:
        """Gets tracks and their features for the Spotify playlist 'New Music Friday'.

        Returns:
            object: the playlist with features as a json
        """
        # collect tracks from global new music friday (nmf) playlist:
        # https://open.spotify.com/playlist/37i9dQZF1DX4JAvHpjipBk?si=a4f193c4d62c4d05
        new_music_friday_playlist_id = "37i9dQZF1DX4JAvHpjipBk"

        try:
            playlist_df = self.get_playlist_with_features(new_music_friday_playlist_id)
        except (ReadTimeoutError, MaxRetryError):
            # in case the https request was not successful
            logging.info("Failed to get the playlist. Trying again after 5 minutes.")
            time.sleep(300)
            playlist_df = self.get_playlist_with_features(new_music_friday_playlist_id)

        return playlist_df.to_json()


if __name__ == "__main__":

    FORMAT = "%(asctime)s|%(levelname)s|%(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    scraper = Scraper()

    app = Flask(__name__)

    # Define API endpoints
    app.add_url_rule(
        "/new_music_friday/", view_func=scraper.get_new_music_friday, methods=["POST"]
    )

    waitress.serve(app, port=8080)
