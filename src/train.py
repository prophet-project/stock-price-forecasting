import tensorflow as tf
import numpy as np
from . import checkpoints
from .save_and_restore import save
from .normalize import datasets
from .model import build_model
from tensorflow.keras.callbacks import CSVLogger

BUFFER_SIZE=64 # Must be grater or equal to batches size
BATCHES=16 # Allow parallel training, but bigger batch may overfit

metrics_file='metrics/training.csv'

print("Tensorflow:", tf.__version__)

# Load normalised datasets
training, validation, testing, input_shape = datasets()
# Dataset data is array of tensors
# if symplify array of tuples: (text: string, label: int)
# where 0 mean bad, and 1 mean good,
# text normalised to input_shape dimension embeed vector


# Build neural network model
model = build_model(input_shape)

train_batches = training.shuffle(BUFFER_SIZE).padded_batch(BATCHES)
validation_batches = validation.padded_batch(BATCHES)

# Train network
model.fit(
        train_batches,
        epochs=10,
        validation_data=validation_batches,
        callbacks=[
            checkpoints.save_weights(), 
            CSVLogger(metrics_file)
        ]
    )

# Save for restore in next time
save(model)
