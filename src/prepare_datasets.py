import os
import pandas as pd
from .libs import params
from .load_datasets import load_datasets
from .window_generator import WindowGenerator
from .indicators import MACD, stochastics_oscillator, ATR

LABEL_SHIFT = params['train']['label_shift']
LABEL_COLUMNS = params['train']['label_columns']

BATCH_SIZE = 33 # divide full dataset on equal batches

feature_list = ['High', 'Low', 'Open', 'Close', 'Volume']

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

    print('full window width =', BATCH_SIZE)
    input_width = round(BATCH_SIZE - LABEL_SHIFT)
    print('input_width =', input_width)

    # make test dataset batches equal size
    test_delimetor = round(len(test_df) / BATCH_SIZE)
    test_df = test_df[:test_delimetor*BATCH_SIZE]
    print('len(test_df)', len(test_df))

    # by some reason last batch len is 1
    train_df = train_df[:-1]
    test_df = test_df[:-1]

    window = WindowGenerator(
        input_width=input_width, label_width=input_width, shift=LABEL_SHIFT,
        train_df=train_df, test_df=test_df,
        label_columns=LABEL_COLUMNS
    )

    print('train shape =', window.example[0].shape)

    return window

def add_indicators(df):
    macd = MACD(df['Close'], 12, 26, 9)
    stochastics = stochastics_oscillator(df['Close'], 14)
    atr = ATR(df, 14)

    df['MACD'] = macd
    df['Stochastics Oscillator'] = stochastics
    df['ATR'] = atr[0]

    return df

"""
    Will normalize datasets and prepare for processing by NN
"""
def build_prepared_dataset():
    train, test = load_datasets()

    train = train[feature_list]
    test = test[feature_list]

    train = add_indicators(train)
    test = add_indicators(test)

    # Normalise data to -1...1
    # for LSTM output range
    train_mean = train.mean()
    train_d = train.max() - train.min()

    train = (train - train_mean) / train_d
    test = (test - train_mean) / train_d

    train = train.dropna()
    test = test.dropna()

    with open(processed_train_dataset_path, 'w') as f:
        train.to_csv(f, index=False)

    with open(processed_test_dataset_path, 'w') as f:
        test.to_csv(f, index=False)    


if __name__ == '__main__':
    build_prepared_dataset()


