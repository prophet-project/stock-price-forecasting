import tensorflow as tf
from normalize import datasets
from save_and_restore import load
import json

print("Tensorflow:", tf.__version__)

metrics_file='metrics/test.json'

# Load normalised datasets
training, validation, testing, input_shape = datasets()

# load model
model = load()

# Test
results = model.evaluate(
    testing.batch(512), 
    verbose=1, 
    callbacks=[]
)

# Save metrics
metrics = {}

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))   
    metrics[name] = value

with open(metrics_file, 'w') as outfile:
    json.dump(metrics, outfile, indent=4)