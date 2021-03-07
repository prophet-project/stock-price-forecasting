import os
import pandas as pd
from .load_datasets import load_datasets
from .window_generator import WindowGenerator
from sklearn.preprocessing import MinMaxScaler

feature_list = ['High', 'Low', 'Open', 'Close']

result_datasets_folder = './preprocessed'
if not os.path.exists(result_datasets_folder):
    os.makedirs(result_datasets_folder)

processed_train_dataset_path = os.path.join(result_datasets_folder, 'training.bitcoin.csv')
processed_test_dataset_path = os.path.join(result_datasets_folder, 'test.bitcoin.csv')

def get_prepared_datasets():
    train = pd.read_csv(processed_train_dataset_path)
    test = pd.read_csv(processed_test_dataset_path)

    return train, test

def make_window_generator(
    input_width, 
    label_width, 
    shift,
    label_columns=None
):
    # Load normalised datasets
    train_df, test_df = get_prepared_datasets()

    window = WindowGenerator(
        input_width=input_width, label_width=label_width, shift=shift,
        train_df=train_df, test_df=test_df,
        label_columns=label_columns
    )

    return window

"""
    Will normalize datasets and prepare for processing by NN
"""
def build_prepared_dataset():
    train, test = load_datasets()

    train = train[feature_list]
    test = test[feature_list]

    # Calc difference for remove trend and normalise

    train_normalised = train.diff()
    test_normalised = test.diff()

    train_normalised.fillna(0, inplace=True)
    test_normalised.fillna(0, inplace=True)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    # fit by only train data, for remove test bias
    scaler.fit(train) 

    train_normalised = pd.DataFrame(scaler.transform(train_normalised), columns=train.columns)
    test_normalised = pd.DataFrame(scaler.transform(test_normalised), columns=test.columns)

    train_normalised.index = train.index
    test_normalised.index = test.index

    with open(processed_train_dataset_path, 'w') as f:
        train.to_csv(f, index=False)

    with open(processed_test_dataset_path, 'w') as f:
        test.to_csv(f, index=False)    


if __name__ == '__main__':
    build_prepared_dataset()


