import tensorflow as tf
from .normalize import datasets
from .save_and_restore import load
import json

print("Tensorflow:", tf.__version__)

# fix issue with "cannot find dnn implementation"
# https://github.com/tensorflow/tensorflow/issues/36508
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True) 
print("Enabled experimental memory growth for", physical_devices[0])

metrics_file='metrics/test.json'

# Load normalised datasets
training, validation, testing, input_shape = datasets()

# load model
model = load()

# Test
results = model.evaluate(
    testing.padded_batch(16), 
    verbose=1,
)

# Save metrics
def save_metrict():
    metrics = {}

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))   
        metrics[name] = value

    with open(metrics_file, 'w') as outfile:
        json.dump(metrics, outfile, indent=4)

save_metrict()