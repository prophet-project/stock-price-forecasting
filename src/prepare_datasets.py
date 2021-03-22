import os
import pandas as pd
import json
from tqdm import tqdm
import talib
from .libs import params, get_from_file, save_to_file
from .load_datasets import load_datasets
from .window_generator import WindowGenerator

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

def make_window_generator():
    # Load normalised datasets
    train_df, test_df = get_prepared_datasets()

    print('len(train_df)', len(train_df))

    print('full window width =', FULL_WINDOW_WITH)
    input_width = round(FULL_WINDOW_WITH - LABEL_SHIFT)
    print('input_width =', input_width)

    # make train dataset batches equal size
    train_delimetor = len(train_df) // (FULL_WINDOW_WITH * BATCH_SIZE)
    train_df = train_df[:train_delimetor*(FULL_WINDOW_WITH * BATCH_SIZE)]

    # make test batches equal size
    test_delimetor = len(test_df) // (FULL_WINDOW_WITH * BATCH_SIZE)
    test_df = test_df[:test_delimetor*(FULL_WINDOW_WITH * BATCH_SIZE)]

    # by some reason last batch is incorrect size
    train_df = train_df[:-7*FULL_WINDOW_WITH]
    test_df = test_df[:-5*FULL_WINDOW_WITH]

    window = WindowGenerator(
        input_width=input_width, label_width=input_width, shift=LABEL_SHIFT,
        train_df=train_df, test_df=test_df,
        label_columns=LABEL_COLUMNS,
        batch_size=BATCH_SIZE
    )

    return window

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

    tqdm.pandas(desc="train dataset")
    train = train.progress_apply(normalize_row, axis=1)
    train.info()
    
    tqdm.pandas(desc="test dataset")
    test = test.progress_apply(normalize_row, axis=1)
    test.info()

    train = train.dropna()
    test = test.dropna()

    with open(processed_train_dataset_path, 'w') as f:
        train.to_csv(f, index=False)

    with open(processed_test_dataset_path, 'w') as f:
        test.to_csv(f, index=False)    


if __name__ == '__main__':
    build_prepared_dataset()


