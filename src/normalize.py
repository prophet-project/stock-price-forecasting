import spacy
import tensorflow as tf
from .datasets import download

# Unfortunately Tensorflow doesn't allow save dataset or tensor in easy way,
# but Spacy process mostly work fast on data processing,
# so we can direcly load datasets and normalaise data each time

VECTOR_SIZE = 300

nlp = spacy.load("en_core_web_lg")

def extract_sentences(text):
    doc = nlp(text)
    return list(doc.sents)

# text = "Peach emoji is where it has always been. Peach is the superior emoji. It's outranking eggplant 🍑 "

def print_sentencies(text):
    for sentence in extract_sentences(text):
        print(sentence)

def token_to_vector(token):
    return token.vector

# normalise text vector of words (vectors of 300 dimension)
def text_to_vector(text):
    doc = nlp(text)

    # map all tokens in sentence to his vectors
    sentence = list(map(token_to_vector, doc))
    # TODO: filter words which out of vocalabirity

    return sentence 

def bytes_to_tensor(bytes):
    text = bytes.numpy().decode("utf-8")
    vector = text_to_vector(text)

    return tf.constant(vector)

def map_func(bytes, label):
    [tensor, ] = tf.py_function(bytes_to_tensor, [bytes], [tf.float32])
    tensor.set_shape([None, VECTOR_SIZE])
    return tensor, label

def normalize_datasets(train, validation, test):
    norm_train = train.map(map_func)
    norm_valid = validation.map(map_func)
    norm_test = test.map(map_func)
    return (norm_train, norm_valid, norm_test, VECTOR_SIZE)

def datasets():
    train_data, validation_data, test_data = download()
    return normalize_datasets(train_data, validation_data, test_data)