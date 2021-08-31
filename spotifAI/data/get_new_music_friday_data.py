"""script that scrapes the 100 newly released tracks from
 the global Spotify playlist 'new music friday' and
 enriches those tracks with additional features
 obtained via the spotify API"""

from dotenv import load_dotenv
import os
import time
from datetime import datetime
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from urllib3.exceptions import ReadTimeoutError, MaxRetryError


load_dotenv()  # load environment variables from .env file (file not on github)


class Scraper:
    def __init__(self):
        cid = os.environ.get("SPOTIFY_CLIENT_ID")
        secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
        client_credentials_manager = SpotifyClientCredentials(
            client_id=cid, client_secret=secret
        )
        self.spotify = spotipy.Spotify(
            client_credentials_manager=client_credentials_manager
        )

    def get_tracks_from_playlist(self, playlist_url):
        """queries data for a specific playlist url"""
        # query tracks of playlist
        track_dicts = self.spotify.playlist(playlist_url)["tracks"]["items"]

        # initialize dataframe
        playlist_df = pd.DataFrame(
            columns=[
                "date_of_scrape",
                "artist",
                "name",
                "track_id",
                "track_popularity",
                "date_added",
                "spotify_url",
            ]
        )

        for index, track_dict in enumerate(track_dicts):

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
            # spotify's popularity is
            popularity = track_dict["popularity"]
            spotify_url = track_dict["external_urls"]["spotify"]

            row_dict = {
                "track_id": [track_id],
                "artist": [track_artist],
                "name": [track_name],
                "track_popularity": [popularity],
                "date_added": [date_added],
                "spotify_url": [spotify_url],
            }

            row_df = pd.DataFrame(row_dict)

            playlist_df = playlist_df.append(row_df)

        playlist_df["date_of_scrape"] = datetime.today().date()

        return playlist_df

    def get_audio_features(self, track_id):
        """returns the audio features for a given track_id"""
        return self.spotify.audio_features(track_id)

    def get_other_track_info(self, track_id):
        """returns the release_date, explicit_boolean and
        a list with artist_ids for a given track_id"""
        track_object = self.spotify.track(track_id)
        release_date = self.get_release_date(track_object)
        explicit_boolean = self.get_explicit_boolean(track_object)
        artist_ids = self.get_artist_ids(track_object)
        return release_date, explicit_boolean, artist_ids

    def get_release_date(self, track_object):
        """returns the release date for a given track_object"""
        return track_object["album"]["release_date"]

    def get_explicit_boolean(self, track_object):
        """returns a boolean value for whether the track contains
        explicit language or not  for a given track_object"""
        return track_object["explicit"]

    def get_artist_ids(self, track_object):
        """returns a list with one or more artist ids for a given track_id"""
        track_artists_object = track_object["artists"]
        return [artist["id"] for artist in track_artists_object]

    def get_artist_features(self, artist_ids):
        """returns the artists' popularity and the number of followers
        for a given list of artist_ids that can contain one or more artists"""
        artists_objects = [self.spotify.artist(uid) for uid in artist_ids]
        followers = self.get_followers(artists_objects)
        artist_popularity = self.get_artist_popularity(artists_objects)
        return followers, artist_popularity

    def get_followers(self, artists_objects):
        """If multiple artists participate in a track,
        the sum of the number of followers of every artist is returned"""
        artists_followers = [artist["followers"]["total"] for artist in artists_objects]
        return sum(artists_followers)

    def get_artist_popularity(self, artists_objects):
        """If multiple artists participate in a track,
        the average of the popularity of every artist is returned,
        rounded to the nearest integer"""
        return round(
            sum([artist["popularity"] for artist in artists_objects])
            / len(artists_objects)
        )


if __name__ == "__main__":

    scraper_object = Scraper()

    def get_new_music_friday(scraper):
        """returns a dataframe with all tracks
        from the global new music friday playlist,
        along with additional features that are
        required for making predictions on hit potential"""
        # collect tracks from global new music friday (nmf) playlist:
        # https://open.spotify.com/playlist/37i9dQZF1DX4JAvHpjipBk?si=a4f193c4d62c4d05
        new_music_friday_playlist_id = "37i9dQZF1DX4JAvHpjipBk"
        nmf = scraper.get_tracks_from_playlist(new_music_friday_playlist_id)

        # collect audio features for the nmf tracks
        audio_features = pd.concat(
            [
                pd.DataFrame(scraper.get_audio_features(track_id=uid))
                for uid in nmf.track_id.values
            ]
        ).rename(columns={"id": "track_id"})

        # merge nmf tracks with audio features
        df = pd.merge(nmf, audio_features, on="track_id")

        # add other track info (release_date, explicit, artist_ids)
        other_track_info_dict = {}
        for track_id in df.track_id.values:
            release_date, explicit_boolean, artist_ids = scraper.get_other_track_info(
                track_id
            )
            other_track_info_dict[track_id] = {
                "release_date": release_date,
                "explicit": explicit_boolean,
                "artist_ids": artist_ids,
            }
        # map values to columns
        df = (
            df.assign(
                release_date=lambda df: df["track_id"].apply(
                    lambda uid: other_track_info_dict.get(uid)["release_date"]
                )
            )
            .assign(
                explicit=lambda df: df["track_id"].apply(
                    lambda uid: other_track_info_dict.get(uid)["explicit"]
                )
            )
            .assign(
                artist_ids=lambda df: df["track_id"].apply(
                    lambda uid: other_track_info_dict.get(uid)["artist_ids"]
                )
            )
        )

        # add artist-level features
        artist_features_dict = {}
        for track_id in df.track_id.values:
            followers, artist_popularity = scraper.get_artist_features(
                df.loc[df.track_id == track_id, "artist_ids"].values[0]
            )
            artist_features_dict[track_id] = {
                "followers": followers,
                "artist_popularity": artist_popularity,
            }
        # map values to columns
        df = df.assign(
            followers=lambda df: df["track_id"].apply(
                lambda uid: artist_features_dict.get(uid)["followers"]
            )
        ).assign(
            artist_popularity=lambda df: df["track_id"].apply(
                lambda uid: artist_features_dict.get(uid)["artist_popularity"]
            )
        )

        print("shape of dataframe with new music friday tracks =", df.shape)
        print("columns = ", df.columns)

        return df

    # scrape tracks from new music friday, along with additional features for modelling
    try:
        df = get_new_music_friday(scraper_object)
    except (ReadTimeoutError, MaxRetryError):
        # in case the https request was not successful
        print("Trying again after 5 minutes")
        time.sleep(300)
        df = get_new_music_friday(scraper_object)
