import os
import pandas as pd
from .load_datasets import load_datasets

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

"""
    Will normalize datasets and prepare for processing by NN
"""
def build_prepared_dataset():
    train, test = load_datasets()

    train = train[feature_list]
    test = test[feature_list]

    # Calc base deviation

    train_mean = train.mean()
    train_std = train.std()

    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std

    with open(processed_train_dataset_path, 'w') as f:
        train.to_csv(f, index=False)

    with open(processed_test_dataset_path, 'w') as f:
        test.to_csv(f, index=False)    


if __name__ == '__main__':
    build_prepared_dataset()


