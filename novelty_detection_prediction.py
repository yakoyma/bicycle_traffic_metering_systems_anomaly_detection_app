"""
===============================================================================
Anomaly Detection Project: Applications dedicated to Novelty Detection for
bicycle traffic metering systems in Nantes
===============================================================================

This file is organised as follows:
1. Load the dataset
2. Feature Engineering
3. Machine Learning
   3.1 Optimisation functions
   3.2 Novelty detection
"""
# Standard libraries
import random
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import numpy as np
import pandas as pd
import sklearn
import pyod
import optuna
import joblib


from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import TargetEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score
from pyod.utils.ad_engine import ADEngine
from pyod.models.lof import LOF
from pyod.models.suod import SUOD
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import evaluate_print
from optuna import create_study
from joblib import dump
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('PyOD: {}'.format(pyod.__version__))
print('Optuna: {}'.format(optuna.__version__))
print('Joblib: {}'.format(joblib.__version__))



# Constants
SEED = 0
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150
FOLDS = 10
CONTAMINATION = 0.05  # Anomalies rate

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Set the maximum number of rows and columns to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)



"""
===============================================================================
1. Load the dataset
===============================================================================
"""
print(f'\n\n\n1. Load the dataset')

# Load the dataset
INPUT_CSV = 'datasets/dataset.csv'
dataset = pd.read_csv(INPUT_CSV, sep=',')



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
print(f'\n\n\n2. Feature Engineering')

# Feature selection
X = dataset.dropna(subset=['Meter reading'])
X = X[X['Anomaly'] == 0]
X = X.dropna(subset=['Meter reading'])
X.reset_index(inplace=True, drop=True)
X = X.drop(['Date', 'Anomaly'], axis=1)
y = np.random.choice(
    [0, 1], size=X.shape[0], p=[1 - CONTAMINATION, CONTAMINATION])

# Classes
print(f'\nNumber of samples per class: {Counter(y)}')
classes = list(set(y))
print(f'Classes: {classes}')

# Create a flag for imputed values of Meter ID feature
X['Meter ID flag'] = False
mask = X['Meter ID'].isna()
X.loc[mask, 'Meter ID flag'] = True

# Imputation of Meter ID feature
mapping = {'La Chapelle sur Erdre': 949, 'Saint Léger les Vignes': 950}
X['Meter ID'] = X['Meter ID'].fillna(X['Meter name'].map(mapping))
X['Meter ID'] = X['Meter ID'].astype(int)
X['Meter ID'] = X['Meter ID'].astype(str)

# Display the head and the tail of the dataset
print(f'\n\nDataset shape: {X.shape}')
print(X.info())
print(pd.concat([X.head(50), X.tail(50)]))


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True, stratify=y)


# Selection of features and the target for regression
X_train_reg = X_train[['Meter name', 'Meter reading', 'Modelled value']]
X_test_reg = X_test[['Meter name', 'Meter reading', 'Modelled value']]
y_train_reg = X_train_reg['Modelled value'].values
y_test_reg = X_test_reg['Modelled value'].values


# Encode the Meter ID and Meter name features for anomaly detection
encoder = TargetEncoder(cv=FOLDS, random_state=SEED)
train_enc = encoder.fit_transform(
    X=X_train[['Meter ID', 'Meter name']], y=y_train)
test_enc = encoder.transform(X=X_test[['Meter ID', 'Meter name']])
X_train['Meter ID'] = train_enc[:, 0]
X_test['Meter ID'] = test_enc[:, 0]
X_train['Meter name'] = train_enc[:, 1]
X_test['Meter name'] = test_enc[:, 1]

# Encoder persistence
dump(encoder, 'models/novelty detection/encoder/encoder.joblib')

# Display the head and the tail of the train set
print(f'\n\nTrain set shape: {X_train.shape}')
print(X_train.info())
print(pd.concat([X_train.head(50), X_train.tail(50)]))

# Display the head and the tail of the test set
print(f'\nTest set shape: {X_test.shape}')
print(X_test.info())
print(pd.concat([X_test.head(50), X_test.tail(50)]))


# Encode the Meter name feature for regression
reg_encoder = TargetEncoder(cv=FOLDS, random_state=SEED)
train_reg_enc = reg_encoder.fit_transform(
    X=X_train_reg['Meter name'].values.reshape(-1, 1), y=y_train_reg)
test_reg_enc = reg_encoder.transform(
    X=X_test_reg['Meter name'].values.reshape(-1, 1))
X_train_reg['Meter name'] = train_reg_enc
X_test_reg['Meter name'] = test_reg_enc

# Encoder persistence
dump(reg_encoder, 'models/novelty detection/encoder/reg_encoder.joblib')


# Normalisation of Meter reading feature for regression
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(
    X=X_train_reg['Meter reading'].values.reshape(-1, 1))
scaled_test = scaler.transform(
    X=X_test_reg['Meter reading'].values.reshape(-1, 1))
X_train_reg['Meter reading'] = scaled_train
X_test_reg['Meter reading'] = scaled_test

# Encoder persistence
dump(scaler, 'models/novelty detection/scaler/scaler.joblib')

# Display the head and the tail of the train set
print(f'\n\nRegression train set shape: {X_train_reg.shape}')
print(X_train_reg.info())
print(pd.concat([X_train_reg.head(50), X_train_reg.tail(50)]))

# Display the head and the tail of the test set
print(f'\nRegression test set shape: {X_test_reg.shape}')
print(X_test_reg.info())
print(pd.concat([X_test_reg.head(50), X_test_reg.tail(50)]))

# Display the head and the tail of the train set
print(f'\n\nRegression train set shape: {X_train_reg.shape}')
print(X_train_reg.info())
print(pd.concat([X_train_reg.head(50), X_train_reg.tail(50)]))

# Display the head and the tail of the test set
print(f'\nRegression test set shape: {X_test_reg.shape}')
print(X_test_reg.info())
print(pd.concat([X_test_reg.head(50), X_test_reg.tail(50)]))


# Correlation analysis
corr_coef = X_train['Meter ID'].corr(X_train['Meter name'])
print(f'\nCorrelation coefficient between Meter ID and Meter name '
      f'features: {corr_coef:.3f}')
corr_coef = X_train_reg['Meter name'].corr(X_train_reg['Meter reading'])
print(f'Correlation coefficient between Meter name and Meter reading '
      f'features: {corr_coef:.3f}')


# Save the training and test datasets in CSV format
X_train_reg.to_csv('datasets/novelty detection/train_dataset.csv', index=False)
X_test_reg.to_csv('datasets/novelty detection/test_dataset.csv', index=False)


# Convert the training and test datasets into arrays
train = X_train.copy()
test = X_test.copy()
X_train = np.array(X_train)
X_test = np.array(X_test)

# Display the head and the tail of the train and the test sets
print(f'\nTrain shape: {np.shape(X_train)}')
print(f'Test shape: {np.shape(X_test)}')

# Display the train and the test labels
print(f'\nTrain label shape: {np.shape(y_train)}')
print(f'Test label shape: {np.shape(y_test)}')



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
print(f'\n\n\n3. Machine Learning')

# 3.1 Optimisation functions
print(f'\n\n3.1 Optimisation functions')

callback = StopOptimisationEarlyCallback(stagnation_threshold=5)
kfolds = StratifiedShuffleSplit(
    n_splits=FOLDS, test_size=0.2, random_state=SEED)


def lof_model_optimisation(trial) -> float:
    """This function performs hyperparameters search for a Novelty Detection
    unsupervised model through cross-validation and returns a score for
    optimisation.

    Args:
        trial (optuna.Trial): an Optuna trial object, used to suggest
                              hyperparameter values

    Returns:
        score (float): the result of model evaluation
    """

    # Instantiate the model
    hyperparams = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 1000),
        'contamination': CONTAMINATION,
        'n_jobs': -1,
        'novelty': True
    }
    model = LOF(**hyperparams)

    # Model optimisation through cross-validation
    scores = list()
    for train_index, val_index in kfolds.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_val_cv = y_train[val_index]

        # Train the model
        model.fit(X_train_cv)

        # Make predictions
        y_proba_val = model.decision_function(X_val_cv)

        # Evaluation
        scores.append(roc_auc_score(y_true=y_val_cv, y_score=y_proba_val))

    score = np.mean(scores)
    return score


def knn_model_optimisation(trial) -> float:
    """This function performs hyperparameters search for a Novelty Detection
    unsupervised model through cross-validation and returns a score for
    optimisation.

    Args:
        trial (optuna.Trial): an Optuna trial object, used to suggest
                              hyperparameter values

    Returns:
        score (float): the result of model evaluation
    """

    # Instantiate the model
    hyperparams = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 1000),
        'n_jobs': -1,
        'contamination': CONTAMINATION
    }
    model = KNN(**hyperparams)

    # Model optimisation through cross-validation
    scores = list()
    for train_index, val_index in kfolds.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_val_cv = y_train[val_index]

        # Train the model
        model.fit(X_train_cv)

        # Make predictions
        y_proba_val = model.decision_function(X_val_cv)

        # Evaluation
        scores.append(roc_auc_score(y_true=y_val_cv, y_score=y_proba_val))

    score = np.mean(scores)
    return score


def iforest_model_optimisation(trial) -> float:
    """This function performs hyperparameters search for a Novelty Detection
    unsupervised model through cross-validation and returns a score for
    optimisation.

    Args:
        trial (optuna.Trial): an Optuna trial object, used to suggest
                              hyperparameter values

    Returns:
        score (float): the result of model evaluation
    """

    # Instantiate the model
    hyperparams = {
        'n_estimators': trial.suggest_int('n_estimators', 1, 1000),
        'contamination': CONTAMINATION,
        'max_features': trial.suggest_float('max_features', 0, 1),
        'n_jobs': -1,
        'random_state': SEED
    }
    model = IForest(**hyperparams)

    # Model optimisation through cross-validation
    scores = list()
    for train_index, val_index in kfolds.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_val_cv = y_train[val_index]

        # Train the model
        model.fit(X_train_cv)

        # Make predictions
        y_proba_val = model.decision_function(X_val_cv)

        # Evaluation
        scores.append(roc_auc_score(y_true=y_val_cv, y_score=y_proba_val))

    score = np.mean(scores)
    return score


# 3.2 Novelty detection
print(f'\n\n3.2 Novelty detection')

# Optimisation using ADEngine model
print(f'\nOptimisation using ADEngine model')

# Instantiate ADEngine model
adengine_model = ADEngine(random_state=SEED)

# Train set profile
train_profile = adengine_model.profile_data(X=train)
print(f'\nTrain set profile: {train_profile}')

# Top 5 models list
models_bench = adengine_model.get_benchmarks(benchmark='all')
print(f"\nTop 5 models list: "
      f"{models_bench['ADBench']['rankings']['overall_top_5']}")

# Find best model
train_plan = adengine_model.plan_detection(
    profile=train_profile, priority='accuracy')
print(f"\nBest model: {train_plan['detector_name']}")
print(f"Result explanation: {train_plan['reason']}")

# Best model explanation
model_info = adengine_model.explain_detector(train_plan['detector_name'])
print(f"\nBest model full name: {model_info['full_name']}")
print(f"This model is best for: {model_info['best_for']}")

# Make predictions
predictions_result = adengine_model.run_detection(
    X_train=train, plan=train_plan, X_test=test)
print(f"\nAnomalies number: {predictions_result['n_anomalies']}")
print(f"Anomalies ratio: {predictions_result['anomaly_ratio']}")
print(f"Runtime (seconds): {predictions_result['runtime_seconds']}")

# Result analysis
result_analysis = adengine_model.analyze_results(
    result=predictions_result, X=train)
print(f"\nPredictions result analysis: {result_analysis['summary']}")

# Result Explanation
result_explanations = adengine_model.explain_findings(
    result=predictions_result, X=train, top_k=5)
print(f"\nPredictions result explanation: {result_analysis['summary']}")
for result_explanation in result_explanations:
    print(f"\nIndex: {result_explanation['index']}")
    print(f"Score: {result_explanation['score']}")
    print(f"Label: {result_explanation['label']}")


# Optimisation of LOF model
lof_study = create_study(direction='maximize')
lof_study.optimize(
    func=lof_model_optimisation,
    n_jobs=-1,
    callbacks=[callback]
)
print(f'\nLOF model best hyperparams: {lof_study.best_params}')

# Optimisation of KNN model
knn_study = create_study(direction='maximize')
knn_study.optimize(
    func=knn_model_optimisation,
    n_jobs=-1,
    callbacks=[callback]
)
print(f'\nKNN model best hyperparams: {knn_study.best_params}')

# Optimisation of IForest model
iforest_study = create_study(direction='maximize')
iforest_study.optimize(
    func=iforest_model_optimisation,
    n_jobs=-1,
    callbacks=[callback]
)
print(f'\nIForest model best hyperparams: {iforest_study.best_params}')

# Hyperparameters
lof_hyperparams = {
    'n_neighbors': lof_study.best_params['n_neighbors'],
    'contamination': CONTAMINATION,
    'n_jobs': -1,
    'novelty': True
}
knn_hyperparams = {
    'n_neighbors': knn_study.best_params['n_neighbors'],
    'n_jobs': -1,
    'contamination': CONTAMINATION
}
iforest_hyperparams = {
    'n_estimators': iforest_study.best_params['n_estimators'],
    'contamination': CONTAMINATION,
    'max_features': iforest_study.best_params['max_features'],
    'n_jobs': -1,
    'random_state': SEED
}
autoencoder_hyperparams = {
    'contamination': CONTAMINATION,
    'random_state': SEED
}
models = [
    KNN(**knn_hyperparams),
    IForest(**iforest_hyperparams),
    AutoEncoder(**autoencoder_hyperparams)
]

# Instantiate the SUOD model
model = SUOD(
    base_estimators=models,
    contamination=CONTAMINATION,
    combination='maximization'
)

# Train the model
model.fit(X_train)

# Make predictions
y_pred = model.predict(X_test)
print(f'\nPredictions shape: {np.shape(y_pred)}')
y_proba = model.decision_function(X_test)
print(f'\nProbabilities shape: {np.shape(y_proba)}')

# Evaluation
evaluate_binary_classification(y_test, y_pred, y_proba, list(set(y_test)))
evaluate_print(clf_name='SUOD', y=y_test, y_pred=y_proba)

# Classes
print(f'\nNumber of samples per class: {Counter(y_pred)}')
classes = list(set(y_pred))
print(f'Classes: {classes}')

# Model persistence
dump(model, 'models/novelty detection/model/model.joblib')
