import pandas as pd
import os
from .libs import params

datasets_folder = './data'
input_dataset = os.path.join(datasets_folder, 'bitstampUSD_1-min_data_2012-01-01_to_2020-12-31.csv')

BATCH_SIZE = params['input']['batch_size']

def load_input_dataset():
    return pd.read_csv(input_dataset)

def load_datasets():
    ...

# Will print dataset sizes
# Not use it in production, 
# size of dataset can be computed only by transformation to list
def print_dataset_sizes(train_data, test_data):
    print(
        '\nLoaded dataset',
        '\ntrain size:', len(list(train_data)),
        '\ntest size:', len(list(test_data)), '\n'
    )
