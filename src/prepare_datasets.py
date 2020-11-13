from .datasets import get_train_dataset_iterator, get_test_dataset_iterator
from .normalize.normalize_text import normalize_text, decode_text_bytes
import pandas as pd
import os
from pathlib import Path
from multiprocessing import Pool

result_datasets_folder = './preprocessed'

processed_train_dataset_path = os.path.join(result_datasets_folder, 'training.1600000.processed.noemoticon.csv')
processed_test_dataset_path = os.path.join(result_datasets_folder, 'testdata.manual.2009.06.14.csv')
        
def process(data):
    (text, label) = data
    text = normalize_text(text)
    return (text, label)

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
                temporal_df = temporal_df.append({'text': text, 'label': label}, ignore_index=True)
                added += 1

                if added >= 100:
                    # flush data to disk
                    temporal_df.to_csv(f, mode='a', header=False)
                    added = 0
                    temporal_df = df

            # flush possible last data
            if added != 0:
                temporal_df.to_csv(f, mode='a', header=False)


# Create file and write header
def create_dataframe(f):
    df = pd.DataFrame({}, columns=['text', 'label'])
    df.to_csv(f)
    return df

def load():
    # TODO
    print('not implemented yet')

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


