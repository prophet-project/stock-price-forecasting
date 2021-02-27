from .datasets import get_train_dataset_iterator, get_test_dataset_iterator, read_dataset_from_csv, wrap_to_tf_dataset, TRAINING_CHUNK_SIZE, TRAINING_CHUNKS_COUNT, TESTING_CHUNK_SIZE, TESTING_CHUNKS_COUNT
from .normalize.normalize_text import normalize_text, decode_text_bytes
import pandas as pd
import os
from pathlib import Path
from multiprocessing import Pool
from progressbar import ProgressBar, UnknownLength, progressbar
import numpy as np

result_datasets_folder = './preprocessed'

processed_train_dataset_path = os.path.join(result_datasets_folder, 'training.1600000.processed.noemoticon.csv')
processed_test_dataset_path = os.path.join(result_datasets_folder, 'testdata.manual.2009.06.14.csv')
        
TEXT_COLUMN='text'
LABEL_COLUMN='label'

def create_data_item(text, label):
    item = {}
    item[TEXT_COLUMN] = text
    item[LABEL_COLUMN] = label
    return item

def process(data):
    (text, label) = data
    processed_text = normalize_text(text)
    if len(processed_text) == 0:
        print('Nothing was returned after normalisation for:', text, ', label:', label)

    return (processed_text, label)

"""
    Will normalize datasets and prepare for processing by NN
"""
def build_prepared_dataset(dataset_iterator, out_file_name):
    with open(out_file_name, 'w') as f:
        df = create_dataframe(f)
        
        with Pool(16) as pool:
            print('Pool created, start processing...')
            results = pool.imap(process, dataset_iterator, 16)

            added = 0
            temporal_df = df
            for (text, label) in results:
                if len(text) > 0:
                    temporal_df = temporal_df.append(create_data_item(text, label), ignore_index=True)
                    added += 1
                else:
                    print("Found empty text after normalisation, will skip")

                if added >= 100:
                    # flush data to disk
                    temporal_df.to_csv(f, mode='a', header=False, index=False)
                    added = 0
                    temporal_df = df

            # flush possible last data
            if added != 0:
                temporal_df.to_csv(f, mode='a', header=False, index=False)


# Create file and write header
def create_dataframe(f):
    df = pd.DataFrame({}, columns=[TEXT_COLUMN, LABEL_COLUMN])
    df.to_csv(f, index=False)
    return df

def get_dataset_iterator(iterator):
    for chunk in iterator:
        for item in chunk.index:
            text = chunk[TEXT_COLUMN][item]
            label = chunk[LABEL_COLUMN][item]

            label = np.float64(label) # for tenserflow compatibility

            yield (text, label)

def get_perprocessed_train_dataset_iterator(display_progress=False):
    iterator = read_dataset_from_csv(processed_train_dataset_path, chunk_size=TRAINING_CHUNK_SIZE)
    if display_progress:
        iterator = progressbar(iterator, max_value=TRAINING_CHUNKS_COUNT, max_error=False)

    return get_dataset_iterator(iterator)

def get_perprocessed_test_dataset_iterator(display_progress=False):
    iterator = read_dataset_from_csv(processed_test_dataset_path, chunk_size=TRAINING_CHUNK_SIZE)
    if display_progress:
        iterator = progressbar(iterator, max_value=TRAINING_CHUNKS_COUNT, max_error=False)

    return get_dataset_iterator(iterator)

def get_perprocessed_train_dataset(display_progress=False):
    generator = lambda: get_perprocessed_train_dataset_iterator(display_progress=display_progress)
    return wrap_to_tf_dataset(generator)

def get_perprocessed_test_dataset(display_progress=False):
    generator = lambda: get_perprocessed_test_dataset_iterator(display_progress=display_progress)
    return wrap_to_tf_dataset(generator) 

def load_preprocessed_datasets(display_train_progress=False, display_test_progress=False):
    train_dataset = get_perprocessed_train_dataset(display_progress=display_train_progress)
    test_dataset = get_perprocessed_test_dataset(display_progress=display_test_progress)

    return train_dataset, test_dataset

if __name__ == '__main__':
    Path(result_datasets_folder).mkdir(parents=True, exist_ok=True)
    build_prepared_dataset(
        get_train_dataset_iterator(display_progress=True), 
        processed_train_dataset_path
    )
    build_prepared_dataset(
        get_test_dataset_iterator(display_progress=True), 
        processed_test_dataset_path
    )


