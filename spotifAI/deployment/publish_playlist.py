"""Script that publishes 'Vantage Hits From The Future' to Spotify.

The top 20 of the new releases from the 'new music friday' playlist
according to the model's predicted ranking is published via the
spotify API.
"""

import logging

import spotipy
import waitress  # type: ignore
from flask import Flask, request
from google.cloud import secretmanager
from spotipy.oauth2 import SpotifyOAuth


class SpotifyPlaylistManager:
    """Class to manage the Spotify playlist."""

    def __init__(self) -> None:
        # authenticate to manage playlist
        scope = "playlist-modify-private"

        cid = self.access_secret_version(
            "projects/420207002838/secrets/SPOTIFY_CLIENT_ID/versions/latest"
        )

        secret = self.access_secret_version(
            "projects/420207002838/secrets/SPOTIFY_CLIENT_SECRET/versions/latest"
        )
        red_uri = self.access_secret_version(
            "projects/420207002838/secrets/SPOTIFY_REDIRECT_URI/versions/latest"
        )

        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=cid,
                client_secret=secret,
                redirect_uri=red_uri,
                scope=scope,
                open_browser=False,
            )
        )

    @staticmethod
    def access_secret_version(secret_version_id: str) -> str:
        """Return the value of a secret's version."""
        # Create the Secret Manager client.
        client = secretmanager.SecretManagerServiceClient()

        # Access the secret version.
        response = client.access_secret_version(name=secret_version_id)

        # Return the decoded payload.
        return response.payload.data.decode("UTF-8")

    def publish_playlist(self) -> str:
        """Publish the playlist to a personal Spotify account.

        Takes in a list with track_ids from the top 20
        of new releases with the highest predicted future
        positions in the charts and publish them to the playlist
        'Vantage Hits From The Future' (by replacing the 20 old
        tracks in there with the 20 new ones).

        The request body should be structured like this:
        {"track_ids":
        ["<track_id_1>","<track_id_2>","<etc>"]}

        Returns:
            - String with information about where to find the playlist
        """
        new_tracks_of_the_week = request.get_json()

        vantage_playlist_id = (
            "https://open.spotify.com/playlist/"
            + "7oCxRqjtXpt5rwwY6nOK4m?si=51de68b4b48c4d88"
        )

        self.sp.playlist_replace_items(
            playlist_id=vantage_playlist_id, items=new_tracks_of_the_week["track_ids"]
        )

        return """Playlist updated! Check out the hits from the future here: \
        https://open.spotify.com/playlist/7oCxRqjtXpt5rwwY6nOK4m?si=51de68b4b48c4d88"""


if __name__ == "__main__":

    FORMAT = "%(asctime)s|%(levelname)s|%(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO)

    playlist_manager = SpotifyPlaylistManager()

    app = Flask(__name__)

    # Define API endpoints
    app.add_url_rule(
        "/publish_playlist/",
        view_func=playlist_manager.publish_playlist,
        methods=["POST"],
    )

    waitress.serve(app, port=8082)
