import tensorflow as tf
from .prepare_datasets import make_window_generator
from .libs import params, prepare, save_metrics, load

prepare(tf)

metrics_file='metrics/test.json'

# Load normalised datasets
window = make_window_generator()

model = load()

# Test
results = model.evaluate(
    window.test, 
    verbose=1,
)

with open(metrics_file, 'w') as outfile:
    save_metrics(model, results, outfile)