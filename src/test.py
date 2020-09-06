import tensorflow as tf
from .normalize import datasets
from .libs import params, prepare, save_metrict, load

prepare(tf)

BATCH_SIZE = params['input']['batch_size']

metrics_file='metrics/test.json'

# For test model need prepare dataset like training dataset,
# except batch size, batches can be another
# Load existing model and train
# For track results better save metrics

# Load normalised datasets
training, validation, testing, input_shape = datasets()

model = load()

# Test
results = model.evaluate(
    testing.padded_batch(BATCH_SIZE), 
    verbose=1,
)

with open(metrics_file, 'w') as outfile:
    save_metrict(model, results, outfile)