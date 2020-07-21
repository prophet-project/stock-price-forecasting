import tensorflow as tf
import datasets
from normalize import normalize_datasets
from save_and_restore import load
import json

print("Tensorflow:", tf.__version__)

metrics_file='test_metrics.json'

# Load dataset
train_data, validation_data, test_data = datasets.download()

# Normalise data
training, validation, testing, input_shape = normalize_datasets(train_data, validation_data, test_data)

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
    json.dump(metrics, outfile)