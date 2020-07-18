import tensorflow_datasets as tfds
import tensorflow as tf

dataset_name = 'imdb_reviews'

# Will return: 
#  20 000 train data
#   5 000 validation data
#  10 000 test data
# in tuples (string, int)
def download():
    train_data, validation_data, test_data = tfds.load(
        name=dataset_name, 
        split=('train[:80%]', 'train[80%:]', 'test'),
        as_supervised=True
    )

    return train_data, validation_data, test_data

# Will print dataset sizes
# Not use it in production, 
# size of dataset can be computed only by transformation to list
def print_dataset_sizes(train_data, validation_data, test_data):
    print(
        '\nLoaded dataset',
        '\ntrain size:', len(list(train_data)), 
        '\nvalidation sise:', len(list(validation_data)),
        '\ntest size:', len(list(test_data)), '\n'
    )

def get_item(dataset, index):
    (text, label) = list(dataset.take(index+1).as_numpy_iterator())[index]
    return text, label


def print_first_items_of_datasets(dataset, normalised):
    # By defaul all data encoded as bytes
    print(get_item(dataset, 3)[0].decode("utf-8"))
    print(get_item(normalised, 3))
