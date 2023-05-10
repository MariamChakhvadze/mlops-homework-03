Homework #3 of the Quantori MLOps course
==============================

This project contains my solutions for homework #3 of the Quantori MLOps course

Description
------------

The homework consists of 10 stages:

1. **Setup**

    Green taxi records from January 2021 to June 2021 were downloaded. Also, the started notebook was obtained. And finally, the repository was created from [the template](https://drivendata.github.io/cookiecutter-data-science/).

2. **Task #1**
   
    [DVC](https://dvc.org/) was installed and "local remote" storage was configured for the entire `data` directory.

3. **Task #2**

    The starter notebook was refactored into several scripts and MLFlow tracking was integrated.

    * [hydra](https://hydra.cc/) support was implemented for configurations.
    * To process data and split it into features and target, `src/features/build_features.py` was implemented. It creates CSV files in `data/processed` directory. Created files were checked using DVC. Also, these files were used in some of the following tasks, especially in training processes. Configuration for this script is in `configs/build_features.yaml`.
    * After processing data, to visualize target distribution, `src/visualization/visualize.py` was created. It takes configurations from `configs/visualize.yaml`, plots the distribution for each target file (can work with several ones) and logs figures in MLFlow `visualizations` experiment.
    * To train the model, `src/models/train_model.py` was implemented. It uses configurations from `configs/train.yaml`. Note, several features/target files can be used for training/validation. The script logs all runs in MLFlow `trainings` experiment with all necessary information, metrics and artifacts. With `log_all_model_params: true` all parameters are logged in MLFlow. It also creates `predictions.txt` (model predictions on validation data) file in `models` directory.

4. **Task #3**

    5 ML models were chosen:

    1. [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) (it was used in the starter notebook)
    2. [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
    3. [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)
    4. [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
    5. [XGBRegressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor)
    
    There is a configuration file for each model in `configs/model`. The model itself can be chosen in `configs/train.yaml`.    

    The training process was executed with all the aforementioned models and inspected in MLFlow UI.
    
5. **Task #4**

    The tuning was implemented for XGBRegressor in `src/models/tune_xgboost.py` using [Optuna](https://optuna.org/). The data is configurable from `configs/tune.yaml`.

6. **Task #5**

    Linear Regression and XGBRegressor were registered in MLFlow model registry using MLFlow UI.
    
7. **Task #6**

    Flask endpoint was created in `src/models/real_time_inference.py`. It serves both registered models. The model can choose using `model_id` parameter (1 - XGBRegressor, 2 - Linear Regression). This endpoint can be tested using `src/models/fake_client.py`.

8. **Task #7**

    Batch prediction script was implemented in `src/models/batch_inference.py`. It is configurable with the model URI and path to data to make predictions on.

9. **Task #8**

    The model were trained again. XGBRegressor was trained on March data and Linear Regression was trained on March and April data. Respectively, April was validation dataset for XGBRegressor and May - for Linear Regression.

    To train/validate model on several files the following configurations can be used:

    ```yaml
    defaults:
    - _self_
    - model: linear_regression

    train_features:
    - data/processed/green_tripdata_2021-03_features.csv
    - data/processed/green_tripdata_2021-04_features.csv
    train_target:
    - data/processed/green_tripdata_2021-03_target.csv
    - data/processed/green_tripdata_2021-04_target.csv
    valid_features: data/processed/green_tripdata_2021-05_features.csv
    valid_target: data/processed/green_tripdata_2021-05_target.csv

    log_all_model_params: false
    ```
   
    The newly trained models were registered in MLFlow model registry using MLFlow UI.
 
    After that offline comparison script was implemented in `src/models/offline_compare.py` which triggers batch predictions on the new month's data (June) using old models and compare against newly trained one. It logs predictions, as well metrics.

10. **Task #9**

    Reporting with [EvidentlyAI](https://www.evidentlyai.com/) was supported in `src/models/make_report.py`. Right now, reports are created on May and June data using XGBRegressor.

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
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
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
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
