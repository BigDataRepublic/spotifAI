# type: ignore
# flake8: noqa
# to-do: refactor into train_model.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:00:56 2021

@author: ridvanz
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    StratifiedKFold,
)
from sklearn.inspection import permutation_importance

import time
import pickle

import lightgbm as lgb

# import optuna
# import optuna.integration.lightgbm as optuna_lgb

import shap

import networkx as nx
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from itertools import permutations
from datetime import datetime

# df = pd.read_csv("/../../spotifAI/data/raw/Database_to_calculate_popularity.csv")

df_full = pd.read_csv(
    "/home/ridvanz/repositories/spotifAI/spotifAI/data/raw/Database_to_calculate_popularity.csv"
)
df_final = pd.read_csv(
    "/home/ridvanz/repositories/spotifAI/spotifAI/data/raw/Final database.csv"
)

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

#
#
#

df_final = df_final.loc[df_final["Country"] == "Global"]
df_full = df_full.loc[df_full["country"] == "Global"]

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

n_songs = len(df_final)

df_full = df_full[["date", "country", "position", "uri"]]

df_full["date"] = pd.to_datetime(df_full["date"])
df_full.dropna(inplace=True)
# df_full['day_of_year'] = df_full['date'].apply(lambda x: x.timetuple().tm_yday)
# variables = variables + ["day_of_year"]

# df_full["score"] =  np.ceil((201 - df_full["position"])**0.5)
df_full["score"] = np.floor(15 * (1 - (df_full["position"] / 200) ** 0.5))

df_full = df_full.set_index("uri").sort_values("date", ascending=True)

# df_full["score"] =  (201 - df_full["position"])

# df_final = df_final.join(np.log(df_full["score"].groupby("uri").sum()))

df = df_final.join(df_full, how="inner")  # , lsuffix='_custom', rsuffix='')

df = df.sort_values(["date", "country"], ascending=True)
# df = df.reset_index().set_index(["idx"])

# df_final.sort_values("score").plot( y ="score")
# df_full["score"].min()
# df_full["score"].hist()
# %%


#%%

# df_final = df_final.sort_values("score", ascending=True)
# X = df_final[variables].astype(np.float32).values
# y = df_final["score"].values.round(0)

# df = df[df["score"]<30]
df_test = df_final.sample(n=0)
df_train = df.drop(index=df_test.index, errors=0)

query_train = df_train.groupby(["date", "country"])["date"].count().to_numpy()

X_train = df_train[variables].astype(np.float32).values
y_train = df_train["score"].values

# X_val = X_train[-1000000:,:]
# y_val = y_train[-1000000:]
# query_val = query_train[-1000000:]

# X_train = X_train[:-1000000,:]
# y_train = y_train[:-1000000]
# query_train = query_train[:-1000000]


# X_test=df_test[variables].astype(np.float32).values

dtrain = lgb.Dataset(X_train, label=y_train, group=query_train, feature_name=variables)
# dval = lgb.Dataset(X_val, label=y_val, group=query_val, feature_name=variables)

#%%


parameters = {
    "objective": "rank_xendcg",
    "metric": "ndcg",
    # 'objective': 'lambdarank',
    "boosting": "goss",
    "num_threads": 4,
    "force_row_wise": True,
    # 'feature_fraction': 0.1,
    # 'bagging_fraction': 0.1,
    # 'bagging_freq': 5
    # "label_gain": list(np.arange(201)**4),
}

tuner = optuna_lgb.LightGBMTunerCV(
    parameters,
    dtrain,
    verbose_eval=0,
    early_stopping_rounds=10,
    nfold=3,
    return_cvbooster=False,
    time_budget=6000,
)

tuner.run()

print("Best score:", tuner.best_score)
best_params = tuner.best_params
print("Best params:", best_params)
print("  Params: ")
for key, value in best_params.items():
    print("{}: {}".format(key, value))
#%% Analyze tuning history
slice_fig = optuna.visualization.plot_slice(tuner.study)
slice_fig.write_html("slice_fig.html", auto_open=True)

history_fig = optuna.visualization.plot_optimization_history(tuner.study)
history_fig.write_html("history_fig.html", auto_open=True)

#%% Use the best parameters found in the previous step to fit the LightGBM on the whole training set
# and finally test the performance on the test set.
params = {
    "objective": "rank_xendcg",
    "metric": "ndcg",
    "boosting": "goss",
    "num_threads": 4,
    "force_row_wise": True,
    "feature_pre_filter": False,
    "lambda_l1": 0,
    "lambda_l2": 0,
    "max_depth": 8,
    "num_leaves": 64,
    "feature_fraction": 1,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "min_child_samples": 20,
}

# # Save the LightGBM model
# with open('lgb_ranker.pkl', 'rb') as f:
#     model= pickle.load(f)

# params= best_params
model = lgb.train(
    params,
    dtrain,
    valid_sets=None,
    verbose_eval=0,
    num_boost_round=100,
    early_stopping_rounds=None,
)
y_pred = model.predict(X_train)

# model.feature_names = variables

# df_test["predicted_ranking"] = y_pred
# df_test = df_test.sort_values("predicted_ranking", ascending=False)
# df_test["popularity"] = np.log(df_test["popularity"])
# df_test.plot.scatter(x="popularity", y="predicted_ranking")

# model.save_model('model.txt', num_iteration=model.best_iteration)
# model = lgb.Booster(model_file='model.txt')


model.feature_names = variables
# Save the LightGBM model
with open("lgb_ranker.pkl", "wb") as f:
    pickle.dump(model, f)

lgb.plot_importance(model)
#%%
from sklearn.metrics import ndcg_score

# load the LightGBM model
with open("lgb_ranker.pkl", "rb") as f:
    modelz = pickle.load(f)

modelz.feature_names


def rank_songs(df, model):

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

    X = df[variables].values
    y = model.predict(X)
    df["rank"] = np.argsort(-y)
    return df


test = rank_songs(df_test, model)
#%% Analyze our final model

# plot builtin feature importance

# Use SHAP library for more advanced model interpretation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)
shap.plots.waterfall(explainer.shap_values(X_test))

# Save the explainer to use in the dashboard
with open("models/shap_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)


shap.plots.beeswarm(shap_values)


#%%
model = lgb.train(
    parameters, dtrain, valid_sets=dval, num_boost_round=2000, early_stopping_rounds=10
)

y_pred = model.predict(X_test)

df_test["predicted_ranking"] = y_pred
df_test = df_test.sort_values("predicted_ranking", ascending=False)


df_test.plot.scatter(x="score", y="predicted_ranking")
# gbm = lgb.LGBMRanker(boosting_type='goss')
# # gbm.fit(X_train, y_train, group=query_train)
# gbm.fit(X_train, y_train, group=query_train,
#         eval_set=[(X_val, y_val)], eval_group=[query_val],
#         eval_at=[5, 10, 20], early_stopping_rounds=50)

# lgb.plot_importance(model)
#%%
edge_weights = df_final["score"].values[None, :] - df_final["score"].values[:, None]

edge_weights[edge_weights < 0] = 0

pairs = np.where(edge_weights != 0)

test = edge_weights[:10, :10]
test = edge_weights[pairs]
feature_df = df_final.set_index("idx")[variables].astype(np.float32)

paired0 = np.concatenate(
    [feature_df.loc[pairs[0]].values, feature_df.loc[pairs[1]].values], axis=1
)
paired1 = np.concatenate(
    [feature_df.loc[pairs[1]].values, feature_df.loc[pairs[0]].values], axis=1
)
paired = np.r_[paired0, paired1]
targets = np.r_[np.ones(len(paired0)), np.zeros(len(paired1))]
weights = np.r_[edge_weights[edge_weights > 0], edge_weights[edge_weights > 0]]
del paired0, paired1

test = paired[:10]

#%%

X_train, X_test, y_train, y_test = train_test_split(
    paired, targets, test_size=0.2, random_state=42, stratify=targets
)
del paired, targets
#%%
dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test)

parameters = {
    "objective": "binary",
    "metric": "auc",
    # 'is_unbalance': 'true',
    # 'boosting': 'gbdt',
    "boosting": "goss",
    # 'num_leaves': 31,
    # 'feature_fraction': 0.5,
    # 'bagging_fraction': 0.1,
    # 'bagging_freq': 1,
    # 'learning_rate': 0.05,
    "num_threads": 4,
    # 'verbose': 0
}

model = lgb.train(
    parameters,
    dtrain,
    valid_sets=dtest,
    num_boost_round=100,
    early_stopping_rounds=None,
)


#%%

y_pred = model.predict(X_test).round(0).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


#%%
edge_weights = np.zeros([n_songs, n_songs])

# indices = [group[1].index for group in df.groupby("date")]
groups = [group[1].position for group in df.groupby("date")]


for k in range(len(groups)):
    rank_diffs = -(groups[k].values[None, :] - groups[k].values[:, None])
    # rank_diffs =  np.sign(rank_diffs)*abs(rank_diffs)
    # rank_diffs =  -np.sign(rank_diffs)
    edge_weights[
        (groups[k].index.values[None, :]), (groups[k].index.values[:, None])
    ] += rank_diffs


edge_weights[edge_weights > 0] = 0
# rank_diffs[rank_diffs<0]= 0
# edge_weights = edge_weights>0

G = nx.from_numpy_matrix(edge_weights, create_using=nx.DiGraph)
# nx.draw(G,with_labels= True)
pr = nx.pagerank_scipy(G, alpha=0.9)
# pr = nx.pagerank(G, alpha=0.9)

#     perms = np.array(list(permutations(groups[k].index,2)))
df_final["pagerank"] = np.log(list(pr.values()))
#     for ind in perms:
#         rank_diff = groups[k][ind[0]] - groups[k][ind[1]]
#         edge_weights[ind[0],ind[1]] += rank_diff
#%%

#%%
df_final["popularity"] = np.log(df_final["popularity"])

df_final.plot.scatter(x="popularity", y="score")
test = df_final.loc["https://open.spotify.com/track/7qiZfU4dY1lWllzX7mPBI3"]

test = df_full.loc["https://open.spotify.com/track/35mvY5S1H3J2QZyna3TFe0"]
(edge_weights == 0).sum()
(edge_weights != 0).sum()

import seaborn as sns

sns.heatmap(edge_weights)

# del df_full
# del df_final
#%%

groups

# %% [markdown]
# ## New data
groups = [
    group[1].drop("date", axis=1).astype(np.float32) for group in df.groupby("date")
]

indices = [group[1].index for group in df.groupby("date")]

test = groups[0].astype(np.float16)
mem = test.memory_usage(deep=True).sum() / 1e6

# df = df.drop("date",axis=1).astype(np.float16)
# %%

X = df_final[variables]
y = df_final["score"]
# randomly split the data
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

# shape of train and test splits
train_x.shape, test_x.shape, train_y.shape, test_y.shape

#%%
forest = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=8, n_jobs=-1)
scores = cross_val_score(
    forest, train_x, train_y, cv=5, scoring="neg_root_mean_squared_error"
)
print(
    "%0.5f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std())
)

# fit the best model with the training data
forest.fit(train_x, train_y)

# predict the target on train and test data
predict_train = forest.predict(train_x)
predict_test = forest.predict(test_x)

# Root Mean Squared Error on train and test data
print(
    "RMSE on train data: ", round(sqrt(mean_squared_error(train_y, predict_train)), 3)
)
print("RMSE on test data: ", round(sqrt(mean_squared_error(test_y, predict_test)), 3))

# Mean Absolute Error on train and test date
print("MAE on train data: ", round(mean_absolute_error(train_y, predict_train), 3))
print("MAE on test data: ", round(mean_absolute_error(test_y, predict_test), 3))

#%%
train_compare = np.stack([train_y.values, predict_train], 1)
test_compare = np.stack([test_y.values, predict_test], 1)

train_compare = train_compare[train_compare[:, 0].argsort()]
test_compare = test_compare[test_compare[:, 0].argsort()]

train_compare = np.c_[train_compare, np.arange(len(train_compare))]
test_compare = np.c_[test_compare, np.arange(len(test_compare))]

train_compare = train_compare[train_compare[:, 1].argsort()]
test_compare = test_compare[test_compare[:, 1].argsort()]

train_compare = np.c_[train_compare, np.arange(len(train_compare))]
test_compare = np.c_[test_compare, np.arange(len(test_compare))]

plt.scatter(train_compare[:, 2], train_compare[:, 3])
plt.scatter(test_compare[:, 2], test_compare[:, 3])

((train_compare[:, 2] - train_compare[:, 3]) ** 2).sum() ** 0.5
((test_compare[:, 2] - test_compare[:, 3]) ** 2).sum() ** 0.5
#%% Calculate and plot feature importances (MDI)

start_time = time.time()
forest.fit(train_x, train_y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

indices = np.argsort(importances)

plt.figure()
plt.title("Feature Importances")
plt.barh(
    range(len(indices)),
    importances[indices],
    color="g",
    ecolor="r",
    align="center",
    xerr=std[indices],
)
plt.yticks(range(len(indices)), train_x.columns[indices])
plt.xlabel("Relative Importance")

# %%

# %% [markdown]
# At first sight, Random Forest Regressor seems to work best.

# %%
# %% [markdown]
# There is some heavy overfitting going on! A problem for later :)

# %%
# pickle model
modelpath = "../../models/"
modelname = "RFregressor_v1.p"
full_modelpath = modelpath + modelname

# add feature names as an attribute to the model to re-use during predictions
model_RF.feature_names = list(train_x.columns.values)

pickle.dump(model_RF, open(full_modelpath, "wb"))

# %% [markdown]
# # simulate predictions on new music friday data

# %%
# load in pickle
loaded_model = pickle.load(open(full_modelpath, "rb"))


# %%
new_data = new_data.assign(
    max_position=[
        round(pred) for pred in loaded_model.predict(new_data[predictor_variables])
    ]
)


# %%
new_data.sort_values(by="max_position")
