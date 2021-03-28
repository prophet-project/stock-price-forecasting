import os
import pandas as pd
import modin.pandas as md
import json
import talib
import tensorflow as tf
from .libs import params, get_from_file, save_to_file
from .load_datasets import load_datasets

LABEL_SHIFT = params['train']['label_shift']
LABEL_COLUMNS = params['train']['label_columns']

BATCH_SIZE = 8 # divide test dataset on equal batches
FULL_WINDOW_WITH = 33
MAX_TARGET = 100000 # maximum expected price

feature_list = ['high', 'low', 'open', 'close', 'volume']

result_datasets_folder = './preprocessed'
if not os.path.exists(result_datasets_folder):
    os.makedirs(result_datasets_folder)

processed_train_dataset_path = os.path.join(result_datasets_folder, 'training.bitcoin.csv')
processed_test_dataset_path = os.path.join(result_datasets_folder, 'test.bitcoin.csv')
normazation_file = os.path.join(result_datasets_folder, 'normalization_params.json')

def get_prepared_datasets():
    train = pd.read_csv(processed_train_dataset_path)
    test = pd.read_csv(processed_test_dataset_path)

    return train, test

def make_generator(data, targets, shuffle, batch_size=BATCH_SIZE, sequence_length=FULL_WINDOW_WITH, sequence_stride=1):
    return tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data[:-sequence_length],
      targets=targets[sequence_length:],
      sequence_length=sequence_length,
      sequence_stride=sequence_stride,
      shuffle=shuffle,
      batch_size=batch_size,
  )


def make_window_generator():
    # Load normalised datasets
    train_df, test_df = get_prepared_datasets()

    train_iterator = make_generator(train_df, train_df[['close']], shuffle=True)
    test_iterator = make_generator(test_df, test_df[['close']], shuffle=False)

    return train_iterator, test_iterator

def add_indicators(df):
    macd, macdsignal, macdhist = talib.MACD(
        df['close'].values, 
        fastperiod=12, slowperiod=26, signalperiod=9
    )
    fastk, fastd = talib.STOCHF(
        df['high'], df['low'], df['close'], 
        fastk_period=14, fastd_period=3
    )
    atr = talib.ATR(
        df['high'], df['low'], df['close'], 
        timeperiod=14
    )

    df['MACD'] = pd.Series(macdhist, index=df.index)
    df['Stochastics Oscillator'] = pd.Series(fastd, index=df.index)
    df['ATR'] = pd.Series(atr, index=df.index)

    return df

# Will be created and saved on building prepared dataset
norm_params = None
norm_d = None

def calc_norm_d():
    global norm_params, norm_d
    dt_max = norm_params['max']
    dt_min = norm_params['min']

    norm_d = dt_max - dt_min

serialisable_params = get_from_file(normazation_file)
if serialisable_params:
    norm_params = {
        'max': pd.Series(serialisable_params['max'], index=serialisable_params['index']),
        'min': pd.Series(serialisable_params['min'], index=serialisable_params['index'])
    }
    calc_norm_d()

def normalize_row(row):
    return row / norm_d

def denormalise_row(row):
    return row * norm_d

"""
    Will normalize datasets and prepare for processing by NN
"""
def build_prepared_dataset():
    global norm_params
    print('Start processing dataset...')
    train, test = load_datasets()

    train = train[feature_list]
    test = test[feature_list]

    train = add_indicators(train)
    test = add_indicators(test)

    # Normalise data to 0...1
    train_max = train.max()
    train_max['high'] = MAX_TARGET
    train_max['low'] = MAX_TARGET
    train_max['open'] = MAX_TARGET
    train_max['close'] = MAX_TARGET

    norm_params = {
        'max': train_max,
        'min': train.min(),
    }
    calc_norm_d()
    print('max:\n', norm_params['max'], '\nmin:\n', norm_params['min'], '\nnorm d:\n', norm_d)
    serialisable_params = {
        'max': norm_params['max'].to_numpy().tolist(),
        'min': norm_params['min'].to_numpy().tolist(),
        'index': norm_params['max'].index.to_numpy().tolist()
    }
    save_to_file(serialisable_params, normazation_file)

    print('processing train dataset...')
    train = md.DataFrame(train)
    train = train.apply(normalize_row, axis=1)
    train.info()
    
    print('processing test dataset...')
    test = md.DataFrame(test)
    test = test.apply(normalize_row, axis=1)
    test.info()

    train = train.dropna()
    test = test.dropna()

    with open(processed_train_dataset_path, 'w') as f:
        train.to_csv(f, index=False)

    with open(processed_test_dataset_path, 'w') as f:
        test.to_csv(f, index=False)    


if __name__ == '__main__':
    build_prepared_dataset()


