import tensorflow as tf
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

def predict(text, model):
    vector = normalize.text_to_vector(text)
    tensor = tf.constant(vector)
    tensor.set_shape([None, normalize.VECTOR_SIZE])
    input_data = tf.data.Dataset.from_tensors([tensor])

    predictions = model.predict(input_data)[0]
    predicted_label = np.argmax(predictions)
    return predicted_label, predictions

