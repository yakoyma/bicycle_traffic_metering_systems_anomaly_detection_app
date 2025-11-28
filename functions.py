"""
===============================================================================
This file contains all the functions for the project
===============================================================================
"""
# Libraries
import matplotlib.pyplot as plt
import numpy as np
import optuna


from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             f1_score,
                             confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay,
                             roc_auc_score,
                             PrecisionRecallDisplay,
                             mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error,
                             median_absolute_error,
                             mean_squared_log_error,
                             max_error,
                             explained_variance_score,
                             r2_score,
                             PredictionErrorDisplay)
from imblearn.metrics import classification_report_imbalanced



def display_pie_chart(dataset, var, figsize):
    """This function displays a pie chart with the proportions and
    count values.

    Args:
        dataset (pd.DataFrame): the Pandas dataset
        var (str): the variable (column of the dataset) to use
        title (str): the title of the chart
        figsize (tuple): the size of the chart
    """

    # Create a series with counted values
    dataviz = dataset[var].value_counts().sort_values(ascending=False)

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('equal')
    ax.pie(
        x=list(dataviz),
        labels=list(dataviz.index),
        autopct='%1.1f%%',
        pctdistance=0.5,
        labeldistance=1.05,
        textprops=dict(color='black', size=12, weight='bold')
    )
    plt.title(f'{var} variable categories', size=18, weight='bold')
    plt.axis('equal')
    plt.grid(False)
    plt.show()


def evaluate_binary_classification(y_test, y_pred, y_proba, labels):
    """This function evaluates the result of a Binary Classification.

    Args:
        y_test (ndarray): the test labels
        y_pred (ndarray): the predicted labels
        y_proba (ndarray): the predicted probabilities
        labels (ndarray): list of unique labels for Confusion Matrix Plot
    """

    print('\n\nAccurcay: {:.3f}'.format(
        accuracy_score(y_true=y_test, y_pred=y_pred)))
    print('Balanced Accurcay: {:.3f}'.format(
        balanced_accuracy_score(y_true=y_test, y_pred=y_pred)))
    print('F1 score: {:.3f}'.format(f1_score(
        y_true=y_test, y_pred=y_pred)))
    print('Confusion Matrix:\n{}'.format(confusion_matrix(
        y_true=y_test, y_pred=y_pred)))
    print('Classification Report:\n{}'.format(classification_report(
        y_true=y_test, y_pred=y_pred)))
    print('Imblearn Classification Report:\n{}'.format(
        classification_report_imbalanced(y_true=y_test, y_pred=y_pred)))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred,
        display_labels=labels,
        xticks_rotation='vertical',
        cmap=plt.cm.Blues
    )
    display.ax_.set_title('Plot of the Confusion Matrix')
    plt.grid(False)
    plt.show()

    if y_proba is not None:
        print('ROC AUC: {:.3f}'.format(roc_auc_score(
            y_true=y_test, y_score=y_proba)))
        display = PrecisionRecallDisplay.from_predictions(
            y_true=y_test, y_pred=y_proba)
        display.ax_.set_title('Precision-Recall curve for test labels')
        plt.grid(True)
        plt.show()


def evaluate_regression(y_test, y_pred, SEED):
    """This function evaluates the result of a Regression.

    Args:
        y_test (ndarray): the test labels
        y_pred (ndarray): the predicted labels
        SEED (ndarray): the random state value
    """

    print('\n\nMSE: {:.3f}'.format(mean_squared_error(y_test, y_pred)))
    print('MAE: {:.3f}'.format(mean_absolute_error(y_test, y_pred)))
    print('MAPE: {:.3f}'.format(mean_absolute_percentage_error(
        y_test, y_pred)))
    print('MdAE: {:.3f}'.format(median_absolute_error(y_test, y_pred)))
    if np.where(y_test < 0)[0].size == 0 and np.where(y_pred < 0)[0].size == 0:
        print('MSLE: {:.3f}'.format(mean_squared_log_error(
            y_test, y_pred)))
    elif np.where(y_test < 0)[0].size > 0:
        print('Impossible to compute MSLE because the test set contains '
              'negative values.')
    elif np.where(y_pred < 0)[0].size > 0:
        print('Impossible to compute MSLE because forecasts or predictions '
              'contain negative values.')
    print('Maximum residual error: {:.3f}'.format(max_error(y_test, y_pred)))
    print('Explained variance score: {:.3f}'.format(explained_variance_score(
        y_test, y_pred)))
    print('R²: {:.3f}'.format(r2_score(y_test, y_pred)))

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred,
        kind='actual_vs_predicted',
        ax=axs[0],
        random_state=SEED
    )
    axs[0].set_title('Actual vs Predicted values')
    axs[0].grid(True)
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred,
        kind='residual_vs_predicted',
        ax=axs[1],
        random_state=SEED
    )
    axs[1].set_title('Residuals vs Predicted Values')
    axs[1].grid(True)
    fig.suptitle('Plot the results of predictions')
    plt.show()


class StopOptimisationEarlyCallback:
    """A callback for Optuna that stops optimisation when there is no
    improvement after a certain number of trials.

    Attributes:
        stagnation_threshold (int): the nuumber of trials without improvement
                                    before stopping
        best_value (float): the best score achieved so far
        stagnation_counter (int): the counter of trials without improvement
    """

    def __init__(self, stagnation_threshold):
        self.stagnation_threshold = stagnation_threshold
        self.best_value = None
        self.stagnation_counter = 0


    def __call__(self, study: optuna.study.Study,
                 trial: optuna.trial.FrozenTrial):

        # If a new better score has been found
        if study.best_trial.number == trial.number:
            self.best_value = study.best_value
            self.stagnation_counter = 0
        else:
            # If best score is already defined
            if self.best_value is not None:
                self.stagnation_counter += 1

                # Stop if stagnation threshold is reached
                if self.stagnation_counter >= self.stagnation_threshold:
                    print(f'\nStopping after {trial.number} trials')
                    print(f'Last improvement at trial: '
                          f'{study.best_trial.number}')
                    print(f'Best score: {study.best_value}')
                    print(f'Best hyperparameters: {study.best_params}')
                    study.stop()
