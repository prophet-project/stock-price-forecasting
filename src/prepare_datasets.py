import os
import pandas as pd
import json
from ta import add_all_ta_features
from ta.utils import dropna
from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from .libs import params, get_from_file, save_to_file
from .load_datasets import load_datasets

LABEL_SHIFT = params['train']['label_shift']
LABEL_COLUMNS = params['train']['label_columns']

BATCH_SIZE = 256
FULL_WINDOW_WITH = 150
MAX_TARGET = 100000 # maximum expected price

feature_list = ['high', 'low', 'open', 'close', 'volume']

result_datasets_folder = './preprocessed'
if not os.path.exists(result_datasets_folder):
    os.makedirs(result_datasets_folder)

processed_train_dataset_path = os.path.join(result_datasets_folder, 'training.bitcoin.csv')
processed_test_dataset_path = os.path.join(result_datasets_folder, 'test.bitcoin.csv')
scaller_file = os.path.join(result_datasets_folder, 'scaler.pkl')

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
    # Dropna from ta also remove zeros and max double value
    df = dropna(df)

    df = add_all_ta_features(
        df, 
        open="open", 
        high="high", 
        low="low", 
        close="close", 
        volume="volume", 
        fillna=True
    )

    df = df[[
         'open', 'high', 'low', 'close', 'volume',
         'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
         'trend_macd', 'momentum_rsi', 'volatility_kchi',
         'trend_ichimoku_conv', 'trend_ichimoku_a', 'trend_ichimoku_b',
         'momentum_stoch', 'momentum_stoch_signal', 'volatility_atr'
    ]]

    return df

def get_scaler():
    with open(scaller_file, 'rb') as f:
        return load(f)

"""
    Will normalize datasets and prepare for processing by NN
"""
def build_prepared_dataset():
    global norm_params
    print('Start processing dataset...')
    train, test = load_datasets()

    train = train[feature_list]
    test = test[feature_list]

    print('Add training indicators...')
    train = add_indicators(train)
    print('Add testing indicators...')
    test = add_indicators(test)

    # Normalise data to 0...1
    train_min = np.min(train)
    train_max = np.max(train)

    train_max['high'] = MAX_TARGET
    train_max['low'] = MAX_TARGET
    train_max['open'] = MAX_TARGET
    train_max['close'] = MAX_TARGET
    train_fit = pd.DataFrame([train_min, train_max])

    scaler = MinMaxScaler()
    scaler = scaler.fit(train_fit)
    print('Save scaller', scaller_file)
    with open(scaller_file, 'wb') as f:
        dump(scaler, f)

    print('processing train dataset...')
    normalised_train = pd.DataFrame(scaler.transform(train))
    normalised_train.columns = train.columns
    normalised_train.index = train.index
    normalised_train.info()
    
    print('processing test dataset...')
    normalised_test = pd.DataFrame(scaler.transform(test))
    normalised_test.columns = test.columns
    normalised_test.index = test.index
    normalised_test.info()

    normalised_train = normalised_train.dropna()
    normalised_test = normalised_test.dropna()

    print('Save processed train dataset', processed_train_dataset_path)
    with open(processed_train_dataset_path, 'w') as f:
        normalised_train.to_csv(f, index=False)

    print('Save processed test dataset', processed_test_dataset_path)
    with open(processed_test_dataset_path, 'w') as f:
        normalised_test.to_csv(f, index=False)    


if __name__ == '__main__':
    build_prepared_dataset()


