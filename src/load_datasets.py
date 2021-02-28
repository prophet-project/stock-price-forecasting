import pandas as pd
import os
from .libs import params

datasets_folder = './data'
input_dataset = os.path.join(datasets_folder, 'coin_Bitcoin.csv')

BATCH_SIZE = params['input']['batch_size']
TRAINING_DATSET_SIZE = params['input']['train_part'] # procent in float
YEARS_COUNT = params['input']['years']

items_in_year = 365
items_count = round(items_in_year * YEARS_COUNT)

def load_input_dataset():
    return pd.read_csv(input_dataset)

def load_datasets():
    input_df = load_input_dataset()
    last_years_dataset = input_df[-1 * items_count:]

    size = len(last_years_dataset)
    train_df = last_years_dataset[0: int(size * TRAINING_DATSET_SIZE)]
    test_df = last_years_dataset[int(size * TRAINING_DATSET_SIZE):]

    return train_df, test_df


# Will print dataset sizes
# Not use it in production, 
# size of dataset can be computed only by transformation to list
def print_dataset_sizes(train_data, test_data):
    print(
        '\nLoaded dataset',
        '\ntrain size:', len(list(train_data)),
        '\ntest size:', len(list(test_data)), '\n'
    )
