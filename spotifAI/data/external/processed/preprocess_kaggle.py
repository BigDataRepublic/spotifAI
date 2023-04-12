"""Script to preprocess, filter and join Kaggle datasets."""

from pathlib import Path

import numpy as np
import pandas as pd

# define file paths
kaggle_dir = Path("SpotifAI/data/external/kaggle")
processed_dir = Path("SpotifAI/data/external/processed")
db_path = kaggle_dir / "Database to calculate popularity.csv"
final_path = kaggle_dir / "Final database.csv"
processed_path = processed_dir / "kaggle.csv"

# read in datasets
df_full = pd.read_csv(db_path)
df_final = pd.read_csv(final_path)

# define variables of interest
variables = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "explicit",
    "followers",
]

# clean and transform datasets
df_full = df_full.loc[df_full["country"] == "Global"]
df_full = df_full[["date", "country", "position", "uri"]]
df_full["date"] = pd.to_datetime(df_full["date"], infer_datetime_format=True)
df_full.dropna(inplace=True)
df_full["score"] = np.floor(15 * (1 - (df_full["position"] / 200) ** 0.5))
df_full = df_full.set_index("uri").sort_values("date", ascending=True)

df_final = df_final.loc[df_final["Country"] == "Global"]
df_final = (
    df_final.rename(columns=str.lower)
    .rename(
        columns={
            "artist_followers": "followers",
            "popu_max": "max_position",
            "acoustics": "acousticness",
            "liveliness": "liveness",
        }
    )[["uri", *variables, "title", "artist", "popularity"]]
    .dropna()
    .drop_duplicates("uri")
    .set_index("uri")
)
df_final[variables] = (
    df_final[variables].apply(pd.to_numeric, errors="coerce", downcast="float").dropna()
)

df = df_final.join(df_full.loc[df_full.index.isin(df_final.index)], how="inner")
df = df.sort_values(["date", "country"], ascending=True)
df.to_csv(processed_path)
