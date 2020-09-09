import spacy
import tensorflow as tf
import tensorflow_datasets as tfds
from .datasets import download

# Unfortunately Tensorflow doesn't allow save dataset or tensor in easy way,
# but Spacy process mostly work fast on data processing,
# so we can direcly load datasets and normalaise data each time

tokenizer = tfds.features.text.Tokenizer()

encoder=None

def build_encoder(labeled_data):
    vocabulary_set = set()
    for text_tensor, _ in labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)

    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    vocab_size = len(vocabulary_set)
    return vocab_size

def print_text(labeled_data, index = 0):
    text = next(iter(labeled_data))[index].numpy()
    print(text)

def text_to_vector(text):
    return encoder.encode(text)

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

    vocab_size = build_encoder(train_data)

    train_data = normalize_dataset(train_data)
    test_data = normalize_dataset(test_data)

    return train_data, test_data, vocab_size