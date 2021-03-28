import tensorflow as tf
from .prepare_datasets import make_window_generator
from .libs import params, prepare, save_metrics, load
from .model import percentage_difference, compile

prepare(tf)

metrics_file='metrics/test.json'

# Load normalised datasets
train, test = make_window_generator()

model = load({"percentage_difference": percentage_difference})
compile(model)

# Test
results = model.evaluate(
    test, 
    verbose=1,
)

with open(metrics_file, 'w') as outfile:
    save_metrics(model, results, outfile)