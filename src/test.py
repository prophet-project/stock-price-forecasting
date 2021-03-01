import tensorflow as tf
from .prepare_datasets import get_prepared_datasets
from .libs import params, prepare, save_metrics, load
from .window_generator import WindowGenerator

prepare(tf)

BATCH_SIZE = params['input']['batch_size']

metrics_file='metrics/test.json'

# Load normalised datasets
train_df, test_df = get_prepared_datasets()

window = WindowGenerator(
        input_width=30, label_width=30, shift=1,
        train_df=train_df, test_df=test_df,
        label_columns=['Close']
    )

model = load()

# Test
results = model.evaluate(
    window.test, 
    verbose=1,
)

with open(metrics_file, 'w') as outfile:
    save_metrics(model, results, outfile)