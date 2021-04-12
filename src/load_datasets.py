import pandas as pd
import os
from .libs import params

datasets_folder = './data'
input_dataset = os.path.join(datasets_folder, '1h_ohlcv.csv')

BATCH_SIZE = params['input']['batch_size']
TRAINING_DATSET_SIZE = params['input']['train_part'] # procent in float


"""will return 5m dataset"""
def load_input_dataset():
    return pd.read_csv(input_dataset)

def split_train_test(df):
    size = len(df)
    train_df = df[0: int(size * TRAINING_DATSET_SIZE)]
    test_df = df[int(size * TRAINING_DATSET_SIZE):]

    return train_df, test_df

def load_datasets():
    df = load_input_dataset()

    return split_train_test(df)


# Will print dataset sizes
# Not use it in production, 
# size of dataset can be computed only by transformation to list
def print_dataset_sizes(train_data, test_data):
    print(
        '\nLoaded dataset',
        '\ntrain size:', len(list(train_data)),
        '\ntest size:', len(list(test_data)), '\n'
    )
