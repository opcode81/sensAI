# Dengue Fever Time Series Prediction with sensAI

This example demonstrates how to use the sensAI library to predict the number of dengue fever cases in San Juan, 
Puerto Rico and Iquitos, Peru on a weekly basis.
The data is a slightly adapted version of the [DengAI dataset](https://www.kaggle.com/datasets/qianyigang129/dengai-dataset/data) from Kaggle, which is licensed under CC0 (public domain).

This example demonstrates
  * time series prediction with sensAI
  * the use of vastly different models with entirely different input pipelines,
    comparing complex dynamic models (that make use the preceding time series)
    with simpler regression models
  * how to use MLflow to record and compare the results of different models
  * hyperparameter optimization with sensAI in conjunction with hyperopt
  * recursive feature elimination

## Setup

Set up your Python virtual environment using [conda](https://docs.anaconda.com/miniconda/miniconda-install/):

    conda env create -f environment.yml

This will create a new environment called `dengai`.

    conda activate dengai

## Entry Points

Initial data analysis with some insights on the dataset can be found in the Jupyter notebook `nb_data_analysis.ipynb`.

Main scripts:
 * `run_model_evaluation.py` to train and evaluate the models
     * Before running it,
        * edit the script and uncomment all the models you want to evaluate.
        * configure the main script at the bottom to configure the concrete task to run.
     * Run the script
     * Inspect results in the `results/` folder or via the MLflow UI (run `mlflow ui`)
 * `run_hyperopt.py` to perform hyperparameter optimization on an XGBoost model
 * `run_rfe.py` to perform recursive feature elimination on an XGBoost model
