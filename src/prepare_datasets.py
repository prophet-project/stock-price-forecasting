import os
import pandas as pd
from .libs import params
from .load_datasets import load_datasets
from .window_generator import WindowGenerator
from .indicators import MACD, stochastics_oscillator, ATR

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
    macd = MACD(df['close'], 12, 26, 9)
    stochastics = stochastics_oscillator(df['close'], 14)
    atr = ATR(df, 14)

    df['MACD'] = macd
    df['Stochastics Oscillator'] = stochastics
    df['ATR'] = atr[0]

    return df

"""
    Will normalize datasets and prepare for processing by NN
"""
def build_prepared_dataset():
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

    train_d = train_max - train.min()

    train = train / train_d
    test = test / train_d

    train = train.dropna()
    test = test.dropna()

    with open(processed_train_dataset_path, 'w') as f:
        train.to_csv(f, index=False)

    with open(processed_test_dataset_path, 'w') as f:
        test.to_csv(f, index=False)    


if __name__ == '__main__':
    build_prepared_dataset()


