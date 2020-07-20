import tensorflow as tf
import numpy as np
import datasets
import checkpoints
from save_and_restore import save
from normalize import normalize_datasets
from model import build_model
from test import test
from tensorflow.keras.callbacks import CSVLogger

BUFFER_SIZE=500 # Must be grater or equal to batches size
BATCHES=256 # Allow parallel training, but bigger batch may overfit

metrics_file='metrics.csv'

print("Tensorflow:", tf.__version__)

# Load dataset
train_data, validation_data, test_data = datasets.download()
# Dataset data is array of tensors
# if symplify array of tuples: (text: string, label: int)
# where 0 mean bad, and 1 mean good

# Normalise data
training, validation, testing, input_shape = normalize_datasets(train_data, validation_data, test_data)

# Build neural network model
model = build_model(input_shape)

# Train network
model.fit(
        training.shuffle(BUFFER_SIZE).batch(BATCHES),
        epochs=10,
        validation_data=validation.batch(BATCHES),
        callbacks=[
            checkpoints.save_weights(), 
            CSVLogger(metrics_file)
        ]
    )

# Test network
test(model, testing)

# Save for restore in next time
save(model)
