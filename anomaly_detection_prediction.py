"""
===============================================================================
Anomaly Detection Project: Applications dedicated to Outlier or Novelty
Detection for bicycle traffic metering systems in Nantes
===============================================================================

This file is organised as follows:
1. Data Analysis
2. Feature Engineering
3. Machine Learning
   3.1 Optimisation functions
   3.2 Novelty detection
   3.3 Outlier detection
       3.3.1 Unsupervised detection
       3.3.2 Supervised detection
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
import sweetviz as sv
import ydata_profiling
import sklearn
import pyod
import optuna
import joblib


from collections import Counter
from sweetviz import analyze
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import TargetEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score
from pyod.models.lof import LOF
from pyod.models.suod import SUOD
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.xgbod import XGBOD
from pyod.utils.data import evaluate_print
from optuna import create_study
from joblib import dump
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Sweetviz: {}'.format(sv.__version__))
print('YData-profiling: {}'.format(ydata_profiling.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('PyOD: {}'.format(pyod.__version__))
print('Optuna: {}'.format(optuna.__version__))
print('Joblib: {}'.format(joblib.__version__))



# Constants
SEED = 0
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150
FOLDS = 10

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Set the maximum number of rows and columns to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)



"""
===============================================================================
1. Data Analysis
===============================================================================
"""
print(f'\n\n\n1. Data Analysis')

# Load the dataset
INPUT_CSV = 'dataset/comptages_velo_nantes_metropole_historique_jour.csv'
raw_dataset = pd.read_csv(INPUT_CSV, sep=';')

# Display the raw dataset's dimensions
print('\n\nDimensions of the raw dataset: {}'.format(raw_dataset.shape))

# Display the raw dataset's information
print('\nInformation about the raw dataset:')
print(raw_dataset.info())

# Description of the raw dataset
print('\nDescription of the raw dataset:')
print(raw_dataset.describe(include='all'))

# Display the head and the tail of the raw dataset
print(f'\nRaw dataset shape: {raw_dataset.shape}')
print(pd.concat([raw_dataset.head(150), raw_dataset.tail(150)]))


# Dispaly the raw dataset report
raw_dataset_report = analyze(source=raw_dataset)
raw_dataset_report.show_html('raw_dataset_report.html')
#report_ydp = ProfileReport(df=raw_dataset, title='Raw Dataset Report')
#report_ydp.to_file('raw_dataset_report_ydp.html')


# Cleanse the dataset
dataset = raw_dataset.rename(
    columns={
        'Identifiant du compteur': 'Meter ID',
        'Jour': 'Date',
        'Nom du compteur': 'Meter name',
        'Anomalie': 'Anomaly',
        'Comptage relevé': 'Meter reading',
        'Valeur modélisée': 'Modelled value'
    }
)
dataset['Anomaly'] = dataset['Anomaly'].fillna(0)
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values(by=['Date'], ascending=True)

# Management of duplicates
print('\n\nManagement of duplicates:')
duplicate = dataset[dataset.duplicated()]
print('Dimensions of the duplicates dataset: {}'.format(duplicate.shape))
print(f'\nDuplicate dataset shape: {duplicate.shape}')
if duplicate.shape[0] > 0:
    dataset = dataset.drop_duplicates()
    dataset.reset_index(inplace=True, drop=True)

# Display the head and the tail of the duplicate
print(f'\nDuplicate shape: {duplicate.shape}')
print(duplicate.info())
print(pd.concat([duplicate.head(150), duplicate.tail(150)]))

# Display the dataset's dimensions
print('\nDimensions of the dataset: {}'.format(dataset.shape))

# Display the dataset's information
print('\nInformation about the dataset:')
print(dataset.info())

# Description of the dataset
print('\nDescription of the dataset:')
print(dataset.describe(include='all'))

# Display the head and the tail of the dataset
print(f'\nDataset shape: {dataset.shape}')
print(pd.concat([dataset.head(150), dataset.tail(150)]))


# Dispaly the dataset report
dataset_report = analyze(source=dataset)
dataset_report.show_html('dataset_report.html')
#dataset_report_ydp = ProfileReport(df=dataset, title='Dataset Report')
#dataset_report_ydp.to_file('dataset_report_ydp.html')


# Visualisations
viz_dataset = dataset.reset_index(drop=True).set_index('Date')
viz_dataset.index = pd.PeriodIndex(viz_dataset.index, freq='D')

# Display the label categories
display_pie_chart(viz_dataset, 'Anomaly', (5, 5))

# Visualisation of temporal trends of Meter reading feature
ax = viz_dataset['Meter reading'].plot(kind='line', figsize=(15, 6))
ax.set_title(f'Temporal trends of Meter reading feature from '
             f'{viz_dataset.index.min()} to {viz_dataset.index.max()}')
ax.set_xlabel('Date')
ax.set_ylabel('Meter reading')
ax.legend(loc='best')
ax.grid(True)
plt.show()

# Visualisation of temporal trends of Modelled value feature
ax = viz_dataset['Modelled value'].plot(kind='line', figsize=(15, 6))
ax.set_title('Temporal trends of modelled value feature from '
             f'{viz_dataset.index.min()} to {viz_dataset.index.max()}')
ax.set_xlabel('Date')
ax.set_ylabel('Modelled value')
ax.legend(loc='best')
ax.grid(True)
plt.show()



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
print(f'\n\n\n2. Feature Engineering')

# Feature selection
# Imputation of the label Anomaly
X = dataset.dropna(subset=['Meter reading'])
X.reset_index(inplace=True, drop=True)
y = X['Anomaly'].values
X = X.drop(['Date', 'Anomaly'], axis=1)

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
print(pd.concat([X.head(150), X.tail(150)]))


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
dump(encoder, 'models/encoder/encoder.joblib')

# Display the head and the tail of the train set
print(f'\n\nTrain set shape: {X_train.shape}')
print(X_train.info())
print(pd.concat([X_train.head(150), X_train.tail(150)]))

# Display the head and the tail of the test set
print(f'\nTest set shape: {X_test.shape}')
print(X_test.info())
print(pd.concat([X_test.head(150), X_test.tail(150)]))


# Encode the Meter name feature for regression
reg_encoder = TargetEncoder(cv=FOLDS, random_state=SEED)
train_reg_enc = reg_encoder.fit_transform(
    X=X_train_reg['Meter name'].values.reshape(-1, 1), y=y_train_reg)
test_reg_enc = reg_encoder.transform(
    X=X_test_reg['Meter name'].values.reshape(-1, 1))
X_train_reg['Meter name'] = train_reg_enc
X_test_reg['Meter name'] = test_reg_enc

# Encoder persistence
dump(reg_encoder, 'models/encoder/reg_encoder.joblib')


# Normalisation of Meter reading feature for regression
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(
    X=X_train_reg['Meter reading'].values.reshape(-1, 1))
scaled_test = scaler.transform(
    X=X_test_reg['Meter reading'].values.reshape(-1, 1))
X_train_reg['Meter reading'] = scaled_train
X_test_reg['Meter reading'] = scaled_test

# Encoder persistence
dump(scaler, 'models/scaler/scaler.joblib')

# Display the head and the tail of the train set
print(f'\n\nRegression train set shape: {X_train_reg.shape}')
print(X_train_reg.info())
print(pd.concat([X_train_reg.head(150), X_train_reg.tail(150)]))

# Display the head and the tail of the test set
print(f'\nRegression test set shape: {X_test_reg.shape}')
print(X_test_reg.info())
print(pd.concat([X_test_reg.head(150), X_test_reg.tail(150)]))

# Display the head and the tail of the train set
print(f'\n\nRegression train set shape: {X_train_reg.shape}')
print(X_train_reg.info())
print(pd.concat([X_train_reg.head(150), X_train_reg.tail(150)]))

# Display the head and the tail of the test set
print(f'\nRegression test set shape: {X_test_reg.shape}')
print(X_test_reg.info())
print(pd.concat([X_test_reg.head(150), X_test_reg.tail(150)]))


# Correlation analysis
corr_coef = X_train['Meter ID'].corr(X_train['Meter name'])
print(f'\nCorrelation coefficient between Meter ID and Meter name '
      f'features: {corr_coef:.3f}')
corr_coef = X_train_reg['Meter name'].corr(X_train_reg['Meter reading'])
print(f'Correlation coefficient between Meter name and Meter reading '
      f'features: {corr_coef:.3f}')


# Save the training and test datasets in CSV format
X_train_reg.to_csv('dataset/train_dataset.csv', index=False)
X_test_reg.to_csv('dataset/test_dataset.csv', index=False)


# Convert the dataframes into arrays
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

# Calculate contamination (rate of outliers)
contamination = Counter(y_train)[1] / y_train.shape[0]
print(f'\n\nContamination: {contamination}')


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
        trial (optuna.Trial): An Optuna trial object, used to suggest
                              hyperparameter values

    Returns:
        score (float): the result of model evaluation
    """

    # Model optimisation through cross-validation
    scores = list()
    for train_index, val_index in kfolds.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_val_cv = y_train[val_index]

        # Instantiate the model
        hyperparams = {
            'n_neighbors': int(trial.suggest_uniform('n_neighbors', 1, 500)),
            'contamination': contamination,
            'n_jobs': -1,
            'novelty': True
        }
        model = LOF(**hyperparams)

        # Train the model
        model.fit(X_train_cv)

        # Make predictions
        y_proba_val = model.decision_function(X_val_cv)

        # Evaluation
        scores.append(roc_auc_score(y_true=y_val_cv, y_score=y_proba_val))

    score = np.mean(scores)
    return score


def knn_model_optimisation(trial) -> float:
    """This function performs hyperparameters search for an Outlier Detection
    unsupervised model through cross-validation and returns a score for
    optimisation.

    Args:
        trial (optuna.Trial): An Optuna trial object, used to suggest
                              hyperparameter values

    Returns:
        score (float): the result of model evaluation
    """

    # Model optimisation through cross-validation
    scores = list()
    for train_index, val_index in kfolds.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_val_cv = y_train[val_index]

        # Instantiate the model
        hyperparams = {
            'n_neighbors': int(trial.suggest_uniform('n_neighbors', 1, 500)),
            'n_jobs': -1,
            'contamination': contamination
        }
        model = KNN(**hyperparams)

        # Train the model
        model.fit(X_train_cv)

        # Make predictions
        y_proba_val = model.decision_function(X_val_cv)

        # Evaluation
        scores.append(roc_auc_score(y_true=y_val_cv, y_score=y_proba_val))

    score = np.mean(scores)
    return score


def iforest_model_optimisation(trial) -> float:
    """This function performs hyperparameters search for an Outlier Detection
    unsupervised model through cross-validation and returns a score for
    optimisation.

    Args:
        trial (optuna.Trial): An Optuna trial object, used to suggest
                              hyperparameter values

    Returns:
        score (float): the result of model evaluation
    """

    # Model optimisation through cross-validation
    scores = list()
    for train_index, val_index in kfolds.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_val_cv = y_train[val_index]

        # Instantiate the model
        hyperparams = {
            'n_estimators': int(trial.suggest_uniform('n_estimators', 1, 500)),
            'contamination': contamination,
            'max_features': trial.suggest_float('max_features', 0, 1),
            'n_jobs': -1,
            'random_state': SEED
        }
        model = IForest(**hyperparams)

        # Train the model
        model.fit(X_train_cv)

        # Make predictions
        y_proba_val = model.decision_function(X_val_cv)

        # Evaluation
        scores.append(roc_auc_score(y_true=y_val_cv, y_score=y_proba_val))

    score = np.mean(scores)
    return score


def xgbod_model_optimisation(trial) -> float:
    """This function performs hyperparameters search for an Outlier Detection
    supervised model through cross-validation and returns a score for
    optimisation.

    Args:
        trial (optuna.Trial): An Optuna trial object, used to suggest
                              hyperparameter values

    Returns:
        score (float): the result of model evaluation
    """

    # Model optimisation through cross-validation
    scores = list()
    for train_index, val_index in kfolds.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

        # Instantiate the model
        hyperparams = {
            'max_depth': int(trial.suggest_uniform('max_depth', 0, 100)),
            'learning_rate': trial.suggest_float('learning_rate', 0, 1),
            'n_estimators': int(trial.suggest_uniform('n_estimators', 1, 800)),
            'n_jobs': -1,
            'gamma': trial.suggest_float('gamma', 0, 10),
            'subsample': trial.suggest_float('subsample', 0, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': SEED
        }
        model = XGBOD(**hyperparams)

        # Train the model
        model.fit(X_train_cv, y_train_cv)

        # Make predictions
        y_proba_val = model.decision_function(X_val_cv)

        # Evaluation
        scores.append(roc_auc_score(y_true=y_val_cv, y_score=y_proba_val))

    score = np.mean(scores)
    return score


# 3.2 Novelty detection
print(f'\n\n3.2 Novelty detection')

# Optimisation of LOF model
lof_study = create_study(direction='maximize')
lof_study.optimize(
    func=lof_model_optimisation,
    n_jobs=-1,
    callbacks=[callback]
)
print(f'\nLOF Best hyperparams:\n{lof_study.best_params}')

# Instantiate the model
lof_hyperparams = {
    'n_neighbors': int(lof_study.best_params['n_neighbors']),
    'contamination': contamination,
    'n_jobs': -1,
    'novelty': True
}
model = LOF(**lof_hyperparams)

# Train the model
model.fit(X_train)

# Make predictions
y_pred = model.predict(X_test)
print(f'\nPredictions shape: {np.shape(y_pred)}')
y_proba = model.decision_function(X_test)
print(f'\nProbabilities shape: {np.shape(y_proba)}')

# Evaluation
evaluate_binary_classification(y_test, y_pred, y_proba, list(set(y_test)))
evaluate_print(clf_name='LOF', y=y_test, y_pred=y_proba)

# Classes
print(f'\nAnomalies classes count: {Counter(y_pred)}')
classes = list(set(y_pred))
print(f'Classes: {classes}')

# Model persistence
dump(model, 'models/novelty detection/model.joblib')


# 3.3 Outlier detection
print(f'\n\n3.3 Outlier detection')

# 3.3.1 Unsupervised detection
print(f'\n\n3.3.1 Unsupervised detection')

# Optimisation of KNN model
knn_study = create_study(direction='maximize')
knn_study.optimize(
    func=knn_model_optimisation,
    n_jobs=-1,
    callbacks=[callback]
)
print(f'\nKNN Best hyperparams:\n{knn_study.best_params}')

# Optimisation of IForest model
iforest_study = create_study(direction='maximize')
iforest_study.optimize(
    func=iforest_model_optimisation,
    n_jobs=-1,
    callbacks=[callback]
)
print(f'\nIForest Best hyperparams:\n{iforest_study.best_params}')

# Hyperparameters
knn_hyperparams = {
    'n_neighbors': int(knn_study.best_params['n_neighbors']),
    'n_jobs': -1,
    'contamination': contamination
}
iforest_hyperparams = {
    'n_estimators': int(iforest_study.best_params['n_estimators']),
    'contamination': contamination,
    'max_features': iforest_study.best_params['max_features'],
    'n_jobs': -1,
    'random_state': SEED
}
autoencoder_hyperparams = {
    'contamination': contamination,
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
    contamination=contamination,
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
dump(model, 'models/outlier detection/unsupervised/model.joblib')


# 3.3.2 Supervised detection
print(f'\n\n3.3.2 Supervised detection')

# Optimisation of XGBOD model
xgbod_study = create_study(direction='maximize')
xgbod_study.optimize(func=xgbod_model_optimisation, n_trials=1, n_jobs=-1)

# Instantiate the model
xgbod_hyperparams = {
    'max_depth': int(xgbod_study.best_params['max_depth']),
    'learning_rate': xgbod_study.best_params['learning_rate'],
    'n_estimators': int(xgbod_study.best_params['n_estimators']),
    'n_jobs': -1,
    'gamma': xgbod_study.best_params['gamma'],
    'subsample': xgbod_study.best_params['subsample'],
    'colsample_bytree': xgbod_study.best_params['colsample_bytree'],
    'reg_alpha': xgbod_study.best_params['reg_alpha'],
    'reg_lambda': xgbod_study.best_params['reg_lambda'],
    'random_state': SEED
}
model = XGBOD(**xgbod_hyperparams)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print(f'\nPredictions shape: {np.shape(y_pred)}')
y_proba = model.decision_function(X_test)
print(f'\nProbabilities shape: {np.shape(y_proba)}')

# Evaluation
evaluate_binary_classification(y_test, y_pred, y_proba, list(set(y_test)))
evaluate_print(clf_name='XGBOD', y=y_test, y_pred=y_proba)

# Classes
print(f'\nNumber of samples per class: {Counter(y_pred)}')
classes = list(set(y_pred))
print(f'Classes: {classes}')

# Model persistence
dump(model, 'models/outlier detection/supervised/model.joblib')
