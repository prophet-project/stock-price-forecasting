from .datasets import get_train_dataset_iterator
from .normalize.normalize_text import normalize_text, decode_text_bytes
import pandas as pd
import os

result_datasets_folder = './data'

processed_train_dataset_path = os.path.join(result_datasets_folder, 'training.1600000.processed.noemoticon.csv')
processed_test_dataset_path = os.path.join(result_datasets_folder, 'testdata.manual.2009.06.14.csv')

"""
    Will normalize datasets and prepare for processing by NN
"""
def build_prepared_dataset():
    train = get_train_dataset_iterator(display_progress=True)

    with open(processed_train_dataset_path, 'w') as f:
        df = create_dataframe(f)
        # TODO: use pool
        for (text, label) in train:
            text = decode_text_bytes(text)
            print('input', text)
            text = normalize_text(text)
            print('result', text)
            append_to_dataframe(df, f, {'text': text, 'label': label})

    # TODO: make same for testing dataset

# Create file and write header
def create_dataframe(f):
    df = pd.DataFrame({}, columns=['text', 'label'])
    df.to_csv(f, index=False)

# create temporal dataframe and append to file
def append_to_dataframe(df, f, dict):
    new_df = df.append(dict, ignore_index=True)
    new_df.to_csv(f, mode='a', header=False)

def load():
    # TODO
    print('not implemented yet')

if __name__ == '__main__':
    build_prepared_dataset()

