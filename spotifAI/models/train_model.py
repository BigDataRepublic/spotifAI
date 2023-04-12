"""Script to train model and upload it to Google Cloud Storage.

The idea is to keep the model simple and focus more on the engineering.
Therefore, no extensive training and testing is performed (yet).
"""

import pickle
from pathlib import Path

import lightgbm as lgb
import optuna.integration.lightgbm as optuna_lgb
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split

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

# read in dataset
processed_dir = Path("SpotifAI/data/external/processed")
df = pd.read_csv(processed_dir / "kaggle.csv")

X = df[variables + ["date"]]
y = df["score"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

query_train = X_train.groupby(["date"])["date"].count().to_numpy()

X_train = X_train.drop("date", axis=1)
X_test = X_test.drop("date", axis=1)

dtrain = lgb.Dataset(X_train, label=y_train, group=query_train, feature_name=variables)

# Perform hyperparameter tuning with Optuna.
# rank_xendcg as a ranking objective is a variant of the traditional cross-entropy loss.
# The ranking objective optimizes the Normalized Discounted Cumulative Gain (NCDG).
# The NDCG metric is a popular metric for evaluating the quality of ranking models.
parameters = {
    "objective": "rank_xendcg",
    "metric": "ndcg",
    "boosting": "gbdt",
    "num_threads": 1,
    "force_row_wise": True,
}

tuner = optuna_lgb.LightGBMTunerCV(
    parameters,
    dtrain,
    verbose_eval=0,
    early_stopping_rounds=3,
    nfold=3,
    return_cvbooster=False,
    time_budget=600,
)

tuner.run()

params = tuner.best_params

lgbm_model = lgb.train(
    params,
    dtrain,
    verbose_eval=10,
    num_boost_round=500,
)

# Save the model to disk in pickle format
VERSION = 2
MODEL_NAME = f"lgbm_model_v{VERSION}.p"

with open(MODEL_NAME, "wb") as f:
    pickle.dump(lgbm_model, f)

# Upload the model to a bucket on Google Cloud Storage
client = storage.Client()
bucket = client.get_bucket("spotifai_bucket")
full_model_path = f"models/{MODEL_NAME}"
blob = bucket.blob(full_model_path)
with open(MODEL_NAME, "rb") as f:
    blob.upload_from_file(f)
print(f"Model uploaded to gs://{bucket.name}/{full_model_path}")
