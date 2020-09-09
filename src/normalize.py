import tensorflow as tf
import tensorflow_datasets as tfds
from .datasets import download
import json

encoder_filename = 'encoder/encoder'
encoder_info_filename ='encoder/info.json'

def load_encoder():
    encoder = tfds.features.text.TokenTextEncoder.load_from_file(encoder_filename)
    vocab_size = None
    
    with open(encoder_info_filename) as info_file:
        info = json.load(info_file)
        vocab_size = info['vocab_size']

    return encoder, vocab_size

# Link to encoder object
# global links not best preactice, but simple enoght
# when this file will be more complex use lambda closure or something like it
CURRENT_ENCODER = None

def print_text(labeled_data, index = 0):
    text = next(iter(labeled_data))[index].numpy()
    print(text)

def text_to_vector(text):
    return CURRENT_ENCODER.encode(text)

def encode(text_tensor, label):
    encoded_text = text_to_vector(text_tensor.numpy())

    return encoded_text, label

def map_func(bytes, label):
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(encode, [bytes, label], Tout=[tf.int64, tf.int64])
    
    # `tf.data.Datasets` work best if all components have a shape set
    #  so set the shapes manually: 
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label

def normalize_dataset(dataset):
    return dataset.map(map_func)

def datasets():
    train_data, test_data = download()
    encoder, vocab_size = load_encoder()

    CURRENT_ENCODER = encoder

    train_data = normalize_dataset(train_data)
    test_data = normalize_dataset(test_data)

    return train_data, test_data, vocab_size