import tensorflow_datasets as tfds
import tensorflow as tf
from .libs import params
import pandas as pd
import os
from progressbar import ProgressBar

# Loaded from http://help.sentiment140.com/for-students
# kaggle copy https://www.kaggle.com/kazanova/sentiment140

datasets_folder = './data'

train_dataset_path = os.path.join(datasets_folder, 'training.1600000.processed.noemoticon.csv')
test_dataset_path = os.path.join(datasets_folder, 'testdata.manual.2009.06.14.csv')

# TODO: split initial train dataset into smallest development dataset (with same distribution of classes)
#  for allow train locally and build smallest vocabulary

LABEL_COLUMN = 'target'
TEXT_COLUMN = 'text'
BATCH_SIZE = params['input']['batch_size']
COLUMNS = ["target", "id", "date", "flag", "user", "text"]
CHUNK_SIZE = 10 ** 3 # Read thousand records at once
TRAINING_RECORDS_COUNT = 1600000
CHUNKS_COUNT = TRAINING_RECORDS_COUNT // CHUNK_SIZE

def get_dataset_iterator(file_path, display_progress=False):
    print('Start reading dataset from', file_path)

    bar = None
    if display_progress:
        bar = ProgressBar(max_value=CHUNKS_COUNT, max_error=False).start()
    
    for i, chunk in enumerate(pd.read_csv(file_path, encoding = "ISO-8859-1", names=COLUMNS, chunksize=CHUNK_SIZE)):
        if bar != None:
            bar.update(i)

        for item in chunk.index:
            text = chunk[TEXT_COLUMN][item]
            label = chunk[LABEL_COLUMN][item]
            
            # Dataset must contain labels in format: 0 = negative, 2 = neutral, 4 = positive
            # but actually conain only "0" and "4", can be string or number
            label = 0 if label == "0" or label == 0 else 1
            
            yield (text, label)

    if bar != None:
        bar.finish()

def get_train_dataset_iterator(display_progress=False):
    return get_dataset_iterator(train_dataset_path, display_progress=display_progress)

def get_test_dataset_iterator(display_progress=False):
    return get_dataset_iterator(test_dataset_path, display_progress=display_progress)

def get_dataset(generator):
    return tf.data.Dataset.from_generator(
        generator, 
        (tf.string, tf.int64), 
        ((), ())
    )

def get_train_dataset(display_progress=False):
    generator = lambda: get_train_dataset_iterator(display_progress=display_progress)
    return get_dataset(generator)

def get_test_dataset(display_progress=False):
    generator = lambda: get_test_dataset_iterator(display_progress=display_progress)
    return get_dataset(generator) 

def download(display_train_progress=False, display_test_progress=False):
    train_dataset = get_train_dataset(display_progress=display_train_progress)
    test_dataset = get_test_dataset(display_progress=display_test_progress)

    return train_dataset, test_dataset

# Will print dataset sizes
# Not use it in production, 
# size of dataset can be computed only by transformation to list
def print_dataset_sizes(train_data, test_data):
    print(
        '\nLoaded dataset',
        '\ntrain size:', len(list(train_data)),
        '\ntest size:', len(list(test_data)), '\n'
    )

def get_item(dataset, index):
    (text, label) = list(dataset.take(index+1).as_numpy_iterator())[index]
    return text, label