[![CI workflow](https://github.com/Vantage-AI/spotifAI/actions/workflows/autoformat_and_lint.yml/badge.svg)](https://github.com/Vantage-AI/spotifAI/actions/workflows/autoformat_and_lint.yml)

spotifAI
==============================

spotify hackathon days

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yml    <- The conda environment file to reproduce the analysis environment. eg.
    │                         `conda env create -f environment.yml`
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── githubrequirements.txt <- Pip references to github repo's, referred to by both `environment.yml` and `requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so spotifAI can be imported
    ├── spotifAI                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes spotifAI a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

## Getting started:

One should be up and running as follows:

    make create_environment
    source activate spotifAI
    make requirements

If you've setup s3, one can sync to and from your bucket as follows:

    make sync_data_from_s3
    make sync_data_to_s3

Running the data preparation should then be as follows:

    make data

### Shortcut creating the environment using conda 
To get started in this project, you first need to setup an environment:

    conda env create -f environment.yml

### Installing Python module as egg
------------
If you want to reuse the code developed in other projects, you can install an egg directly from your checkout:

    pip install -e .


<p><small>Project based on the <a target="_blank" href="https://github.com/BigDataRepublic/cookiecutter-data-science">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
