"""
===============================================================================
Anomaly Detection Project: Applications dedicated to Outlier or Novelty
Detection for bicycle traffic metering systems in Nantes - Regression to
predict the feature Modelled value
===============================================================================
"""
# Standard libraries
import random
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hyperopt
import sklearn
import optuna
import joblib


from hpsklearn import (HyperoptEstimator, any_regressor)
from hyperopt import tpe
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from perpetual import PerpetualBooster
from optuna import create_study
from joblib import dump
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Hyperopt: {}'.format(hyperopt.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('Optuna: {}'.format(optuna.__version__))
print('Joblib: {}'.format(joblib.__version__))



# Constants
SEED = 0
FOLDS = 10

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)



def load_datasets(train_path, test_path):
    """This function loads CSV files into a Pandas DataFrame containing the
    training and test datasets, then converts them into features and targets.

    Args:
        train_path (str): the csv file path for the train dataset
        test_path (str): the csv file path for the test dataset

    Returns:
        X_train (ndarray): the train set
        X_test (ndarray): the test set
        y_train (ndarray): the train target
        y_test (ndarray): the test target
    """

    # Load train and test sets
    train_dataset = pd.read_csv(train_path, sep=',')
    test_dataset = pd.read_csv(test_path, sep=',')
    y_train = train_dataset['Modelled value'].values
    y_test = test_dataset['Modelled value'].values
    X_train = np.array(train_dataset.drop(['Modelled value'], axis=1))
    X_test = np.array(test_dataset.drop(['Modelled value'], axis=1))
    return X_train, X_test, y_train, y_test


def budget_hyperparameter_optimisation(trial) -> float:
    """This function performs the search for the best value of the budget
    hyperparameter for the PerpetualBooster regression model and returns a
    score for optimisation.

    Args:
        trial (optuna.Trial): An Optuna trial object, used to suggest
                              hyperparameter values

    Returns:
        score (float): the result of model evaluation
    """

    # Model optimisation through cross-validation
    scores = list()
    kfolds = ShuffleSplit(n_splits=FOLDS, test_size=0.2, random_state=SEED)
    for train_index, val_index in kfolds.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

        # Instantiate the model
        hyperparams = {
            'objective': 'SquaredLoss',
            'budget': trial.suggest_float('budget', 0, 2.1)
        }
        model = PerpetualBooster(**hyperparams)

        # Train the model
        model.fit(X_train_cv, y_train_cv)

        # Make predictions
        y_pred_val = model.predict(X_val_cv)

        # Evaluation
        scores.append(mean_squared_error(y_true=y_val_cv, y_pred=y_pred_val))

    score = np.mean(scores)
    return score



if __name__ == "__main__":

    # Load the training and test sets (features and targets)
    TRAIN_INPUT_CSV = 'dataset/train_dataset.csv'
    TEST_INPUT_CSV = 'dataset/test_dataset.csv'
    X_train, X_test, y_train, y_test = load_datasets(
        TRAIN_INPUT_CSV, TEST_INPUT_CSV)


    # 1.1 Hyperopt-sklearn
    print('\n\n1.1 Hyperopt-sklearn')
    
    # Instantiate the HyperoptEstimator
    estimators = HyperoptEstimator(
        regressor=any_regressor('reg'),
        algo=tpe.suggest,
        seed=SEED,
        verbose=True,
        n_jobs=-1
    )

    # Train the model
    estimators.fit(X_train, y_train)

    # Display information about the best model
    model = estimators.best_model()['learner']
    print('\nBest model:\n{}'.format(model))

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluation
    evaluate_regression(y_test, y_pred, SEED)

    # Model persistence
    dump(model, 'models/regression/hpsklearn/model.joblib')


    # 1.2 Perpetual ML
    print('\n\n1.2 Perpetual ML')

    # Optimisation of the model
    callback = StopOptimisationEarlyCallback(stagnation_threshold=5)
    study = create_study(direction='minimize')
    study.optimize(
        func=budget_hyperparameter_optimisation,
        n_jobs=-1,
        callbacks=[callback]
    )
    budget = study.best_params['budget']
    print('\nBudget: {}'.format(budget))

    # Instantiate the model
    model = PerpetualBooster(objective='SquaredLoss', budget=budget)

    # Train the model
    model.fit(X_train, y_train)

    # Train score
    print('\nTrain score:\n{}'.format(model.base_score))

    # Make predictions
    y_pred = model.predict(X_test)

    # Best hyperparameters of the model
    print('\nBest hyperparameters:\n{}'.format(model.get_params()))

    # Evaluation
    evaluate_regression(y_test, y_pred, SEED)

    # Feature importance
    print('\nTotal gain:\n{}'.format(
        model.calculate_feature_importance('TotalGain')))
    print('Gain:\n{}'.format(model.calculate_feature_importance('Gain')))

    # Model persistence
    dump(model, 'models/regression/perpetualbooster/model.joblib')
