import tensorflow as tf
from .prepare_datasets import make_window_generator
from .libs import params, prepare, save_metrics, load, checkpoints
from .model import build_model

prepare(tf)

metrics_file='metrics/test.json'

# Load normalised datasets
train, test = make_window_generator()

model = build_model()
model = checkpoints.load_weights(model)

# Test
results = model.evaluate(
    test, 
    verbose=1,
)

with open(metrics_file, 'w') as outfile:
    save_metrics(model, results, outfile)