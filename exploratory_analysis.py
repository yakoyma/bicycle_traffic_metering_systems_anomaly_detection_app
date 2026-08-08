"""
===============================================================================
Anomaly Detection Project: Applications dedicated to Outlier or Novelty
Detection for bicycle traffic metering systems in Nantes
===============================================================================

This file is organised as follows:
1. Load the dataset
2. Cleanse the dataset
3. Save the cleanse dataset
"""
# Standard libraries
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sweetviz as sv
import ydata_profiling


from collections import Counter
from sweetviz import analyze
from ydata_profiling import ProfileReport
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Sweetviz: {}'.format(sv.__version__))
print('YData-profiling: {}'.format(ydata_profiling.__version__))



# Constants
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150

# Set the maximum number of rows and columns to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)



"""
===============================================================================
1. Load the dataset
===============================================================================
"""
print(f'\n\n\n1. Load the dataset')

# Load the raw dataset
INPUT_CSV = 'datasets/comptages_velo_nantes_metropole_historique_jour.csv'
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
print(pd.concat([raw_dataset.head(50), raw_dataset.tail(50)]))


# Display the raw dataset report
raw_dataset_report = analyze(source=raw_dataset)
raw_dataset_report.show_html('raw_dataset_report.html')
#report_ydp = ProfileReport(df=raw_dataset, title='Raw Dataset Report')
#report_ydp.to_file('raw_dataset_report_ydp.html')



"""
===============================================================================
2. Cleanse the dataset
===============================================================================
"""
print(f'\n\n\n2. Cleanse the dataset')

# Cleanse the raw dataset
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
print(pd.concat([duplicate.head(50), duplicate.tail(50)]))

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
print(pd.concat([dataset.head(50), dataset.tail(50)]))


# Display the dataset report
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
3. Save the cleanse dataset
===============================================================================
"""
print(f'\n\n\n3. Save the cleanse dataset')

# Save the training and test datasets in CSV format
dataset.to_csv('datasets/dataset.csv', index=False)
