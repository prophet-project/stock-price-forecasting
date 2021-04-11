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
from .load_datasets import load_input_dataset, split_train_test

LABEL_SHIFT = params['train']['label_shift']
LABEL_COLUMNS = params['train']['label_columns']

BATCH_SIZE = 256
FULL_WINDOW_WITH = 150
MAX_TARGET = 100000 # maximum expected price

feature_list = ['high', 'low', 'open', 'close', 'volume']

result_datasets_folder = './preprocessed'
if not os.path.exists(result_datasets_folder):
    os.makedirs(result_datasets_folder)

full_features_dataset = os.path.join(result_datasets_folder, 'full_featutes.5m.bitcoin.csv')
processed_train_dataset_path = os.path.join(result_datasets_folder, 'training.5m.bitcoin.csv')
processed_test_dataset_path = os.path.join(result_datasets_folder, 'test.5m.bitcoin.csv')
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

def add_all_indicators(df):
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

    return df

def get_important_fetures(df):
    df = df[[
        'open', 'high', 'low', 'close', 'volume',
        'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
        'trend_macd', 'momentum_rsi', 'volatility_kchi',
        'trend_ichimoku_conv', 'trend_ichimoku_a', 'trend_ichimoku_b',
        'momentum_stoch', 'momentum_stoch_signal', 'volatility_atr'
    ]]

    return df

def add_indicators(df):

    df = add_all_indicators(df)
    df = get_important_fetures(df)

    return df

def get_scaler():
    with open(scaller_file, 'rb') as f:
        return load(f)

def get_full_features_dataset():
    if os.path.exists(full_features_dataset):
        print('full features dataset already exists, will read from:', full_features_dataset)
        return pd.read_csv(full_features_dataset)

    print('Start processing dataset...')
    df = load_input_dataset()

    print('Add all indicators...')
    df = add_all_indicators(df)

    print('Save full feature dataset to:', full_features_dataset)
    with open(full_features_dataset, 'w') as f:
        df.to_csv(f)
    
    return df

def fit_scaler(train):
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

    return scaler

def normalize_dataset(df, scaler):
    normalised = pd.DataFrame(scaler.transform(df))
    normalised.columns = df.columns
    normalised.index = df.index
    
    return normalised

"""
    Will normalize datasets and prepare for processing by NN
"""
def build_prepared_dataset():
    df = get_full_features_dataset()

    df.index = pd.to_datetime(df.pop('timestamp'), unit='ms')
    df = get_important_fetures(df)

    train, test = split_train_test(df)

    print('fit scaller...')
    scaler = fit_scaler(train)

    print('normalizing train dataset...')
    train = normalize_dataset(train, scaler)
    train = train.dropna()
    train.info()

    print('Save processed train dataset', processed_train_dataset_path)
    with open(processed_train_dataset_path, 'w') as f:
        train.to_csv(f)
    
    print('normalizing test dataset...')
    test = normalize_dataset(test, scaler)
    test = test.dropna()
    test.info()

    print('Save processed test dataset', processed_test_dataset_path)
    with open(processed_test_dataset_path, 'w') as f:
        test.to_csv(f)    


if __name__ == '__main__':
    build_prepared_dataset()


