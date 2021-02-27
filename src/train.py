import tensorflow as tf
from .libs import params, prepare, save, checkpoints
from .prepare_datasets import get_prepared_datasets
from .model import build_model
from tensorflow.keras.callbacks import CSVLogger

prepare(tf)

BUFFER_SIZE = params['train']['buffer_size'] 
BATCH_SIZE = params['input']['batch_size']
EPOCHS = params['train']['epochs']

metrics_file='metrics/training.csv'

# Load normalised datasets
training, testing = get_prepared_datasets()
# Dataset data is array of tensors
# if symplify array of tuples: (text: string, label: int)
# where 0 mean bad, and 1 mean good

# Build neural network model
model = build_model()

train_batches = training.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
validation_batches = testing.padded_batch(BATCH_SIZE)

# Train network
model.fit(
    train_batches,
    epochs=EPOCHS,
    validation_data=validation_batches,
    callbacks=[
        checkpoints.save_weights(), 
        CSVLogger(metrics_file)
    ]
)

# Save for restore in next time
save(model)
