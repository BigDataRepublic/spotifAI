"""script that publishes the top 20 of the
new releases from the 'new music friday' playlist
ranked via the spotify API"""

from dotenv import load_dotenv
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import waitress
from flask import Flask, request

load_dotenv()  # load environment variables from .env file (file not on github)


class SpotifyPlaylistManager:
    """Class to programmatically control the content
    of our spotify playlist 'Vantage Hits From The Future'"""

    def __init__(self):
        # authenticate to manage playlist
        scope = "playlist-modify-private"

        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
                client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET"),
                redirect_uri=os.environ.get("SPOTIFY_REDIRECT_URI"),
                scope=scope,
            )
        )

    def publish_playlist(self):
        """Takes in a list with track_ids from the top 20
        of new releases with the highest predicted future
        positions in the charts and publish them to the playlist
        "Vantage Hits From The Future" (by replacing the 20 old
         tracks in there with the 20 new ones)

         The request body should be structured like this:
         {"track_ids":
         ["<track_id_1>","<track_id_2>","<etc>"]}
        """

        new_tracks_of_the_week = request.get_json()

        vantage_playlist_id = (
            "https://open.spotify.com/playlist/"
            + "7oCxRqjtXpt5rwwY6nOK4m?si=51de68b4b48c4d88"
        )

        self.sp.playlist_replace_items(
            playlist_id=vantage_playlist_id, items=new_tracks_of_the_week["track_ids"],
        )

        return """Playlist updated! Check out the hits from the future here: \
        https://open.spotify.com/playlist/7oCxRqjtXpt5rwwY6nOK4m?si=51de68b4b48c4d88"""


if __name__ == "__main__":

    playlist_manager = SpotifyPlaylistManager()

    app = Flask(__name__)

    # Define API endpoints
    app.add_url_rule(
        "/publish_playlist/",
        view_func=playlist_manager.publish_playlist,
        methods=["POST"],
    )

    waitress.serve(app, port=8081)
