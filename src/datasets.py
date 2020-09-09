import tensorflow_datasets as tfds
import tensorflow as tf
from .libs import params
import pandas as pd
import os

# Loaded from http://help.sentiment140.com/for-students
# kaggle copy https://www.kaggle.com/kazanova/sentiment140

datasets_folder = './data'

train_dataset_path = os.path.join(datasets_folder, 'training.1600000.processed.noemoticon.csv')
test_dataset_path = os.path.join(datasets_folder, 'testdata.manual.2009.06.14.csv')

LABEL_COLUMN = 'target'
TEXT_COLUMN = 'text'
BATCH_SIZE = params['input']['batch_size']
COLUMNS = ["target", "id", "date", "flag", "user", "text"]

def get_dataset(file_path):
    df = pd.read_csv(file_path, encoding = "ISO-8859-1", names=COLUMNS)
    
    df[LABEL_COLUMN] = pd.Categorical(df[LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].cat.codes

    labels = df.pop(LABEL_COLUMN)
    texts = df.pop(TEXT_COLUMN)

    return tf.data.Dataset.from_tensor_slices((texts.values, labels.values))

def download():
    train_dataset = get_dataset(train_dataset_path)
    test_dataset = get_dataset(test_dataset_path)

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