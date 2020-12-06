import tensorflow as tf
import tensorflow_datasets as tfds
from .prepare_datasets import get_perprocessed_train_dataset, load_preprocessed_datasets
import json

# Link to encoder object
# global links not best preactice, but simple enoght
# when this file will be more complex use lambda closure or something like it
CURRENT_ENCODER = None

encoder_filename = 'encoder/encoder'
encoder_info_filename ='metrics/encoder.json'

# Need firstly build encoder on training dataset, 
# for embeding of all text input

def build_encoder(labeled_data):
    vocabulary_set = set()
    for text_tensor, _ in labeled_data:
        text_tokens = text_tensor.numpy()
        vocabulary_set.update(text_tokens)

    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    vocab_size = len(vocabulary_set)
    return encoder, vocab_size

def text_to_vector(text):
    preprocessed = preprocess_text(text)
    return CURRENT_ENCODER.encode(' '.join(preprocessed))

def encode(text_tensor, label):
    encoded_text = text_to_vector(text_tensor.numpy())

    return encoded_text, label

def text_bytes_to_vectors(bytes, label):
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(encode, [bytes, label], Tout=[tf.int64, tf.int64])
    
    # `tf.data.Datasets` work best if all components have a shape set
    #  so set the shapes manually: 
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label

def vectorise_dataset(dataset):
    return dataset.map(map_func)

def save(encoder, info_data):
    encoder.save_to_file(encoder_filename)
    print("Encoder saved to", encoder_filename)

    with open(encoder_info_filename, 'w') as info:
        json.dump(info_data, info)
    
    print('Encoder info saved to', encoder_info_filename)

def load():
    encoder = tfds.features.text.TokenTextEncoder.load_from_file(encoder_filename)
    vocab_size = None
    
    with open(encoder_info_filename) as info_file:
        info = json.load(info_file)
        vocab_size = info['vocab_size']

    return encoder, vocab_size

def get_vectorised_datasets():
    train_data, test_data = load_preprocessed_datasets()
    encoder, vocab_size = load_encoder()

    CURRENT_ENCODER = encoder

    train_data = vectorise_dataset(train_data)
    test_data = vectorise_dataset(test_data)

    return train_data, test_data, vocab_size

if __name__ == '__main__':
    print('Start building encoder...')

    train_data = get_perprocessed_train_dataset(display_progress=True)
    encoder, vocab_size = build_encoder(train_data)

    print('Encoder was build, saving...')
    save(encoder, { 'vocab_size': vocab_size })