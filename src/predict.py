import tensorflow as tf
import datasets
from normalize import normalize_datasets
from save_and_restore import load
from model import build_model

print("Tensorflow:", tf.__version__)

# Load dataset
train_data, validation_data, test_data = datasets.download()

# Normalise data
training, validation, testing, input_shape = normalize_datasets(train_data, validation_data, test_data)

# load model
model = load()

# print first items of datasets
# By defaul all data encoded as bytes
print(datasets.get_item(train_data, 3)[0].decode("utf-8"))
# print(datasets.get_item(training, 3))

vector_of_text, label =  datasets.get_item(training, 3)
input_data = tf.data.Dataset.from_tensors([vector_of_text])

# Predict
[predicted_label] = model.predict(input_data).flatten()

print('Predicted label:', predicted_label, 'real label: ', label)
if (predicted_label == label):
    print('Successfully predicted')
else:
    print('Failed to predict')