import tensorflow as tf
from .prepare_datasets import get_prepared_datasets
from .libs import params, prepare, save_metrics, load_custom
from .window_generator import WindowGenerator
from .model import build_model

prepare(tf)

BATCH_SIZE = params['input']['batch_size']
LABEL_STEPS = params['train']['label_steps']
INPUT_WIDTH = params['train']['input_width']

metrics_file='metrics/test.json'

# Load normalised datasets
train_df, test_df = get_prepared_datasets()

window = WindowGenerator(
    input_width=INPUT_WIDTH, label_width=LABEL_STEPS, shift=LABEL_STEPS,
    train_df=train_df, test_df=test_df,
    label_columns=['Close']
)

model = build_model()
load_custom(model)

# Test
results = model.evaluate(
    window.test, 
    verbose=1,
)

with open(metrics_file, 'w') as outfile:
    save_metrics(model, results, outfile)