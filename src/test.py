import tensorflow as tf
from .prepare_datasets import make_window_generator
from .libs import params, prepare, save_metrics, load

prepare(tf)

INPUT_WIDTH = params['train']['input_width']
LABEL_STEPS = params['train']['label_width']
LABEL_SHIFT = params['train']['label_shift']
LABEL_COLUMNS = params['train']['label_columns']

metrics_file='metrics/test.json'

# Load normalised datasets
window = make_window_generator(
    input_width=INPUT_WIDTH, label_width=LABEL_STEPS, shift=LABEL_SHIFT, 
    label_columns=LABEL_COLUMNS
)

model = load()

# Test
results = model.evaluate(
    window.test, 
    verbose=1,
)

with open(metrics_file, 'w') as outfile:
    save_metrics(model, results, outfile)