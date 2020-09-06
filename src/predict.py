import tensorflow as tf
from . import datasets
from . import normalize
from .libs import load
import numpy as np

def get_probability_model():
    # Add probablity layer for easier understand class
    model = load()
    probability_model = tf.keras.Sequential([
        model, 
        # Convert logits to predictions
        tf.keras.layers.Softmax()
    ])
    return probability_model

def get_text_and_label_from_dataset(index):
    # Load dataset
    train_data, validation_data, test_data = datasets.download()
    encoded_text, label = datasets.get_item(train_data, index)
    # all text data encoded as bytes
    text = encoded_text.decode('utf-8')
    return text, label

def predict(text, model):
    vector = normalize.text_to_vector(text)
    tensor = tf.constant(vector)
    tensor.set_shape([None, normalize.VECTOR_SIZE])
    input_data = tf.data.Dataset.from_tensors([tensor])

    predictions = model.predict(input_data)[0]
    predicted_label = np.argmax(predictions)
    return predicted_label, predictions

