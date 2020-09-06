import tensorflow as tf
import numpy as np
import yaml
from . import checkpoints
from .save_and_restore import save
from .normalize import datasets
from .model import build_model
from tensorflow.keras.callbacks import CSVLogger

# fix issue with "cannot find dnn implementation"
# https://github.com/tensorflow/tensorflow/issues/36508
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True) 
print("Enabled experimental memory growth for", physical_devices[0])

BUFFER_SIZE=None 
BATCH_SIZE=None

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)
    BUFFER_SIZE = params['train']['buffer_size']
    BATCH_SIZE = params['input']['batch_size']
    print('Params: BUFFER_SIZE =', BUFFER_SIZE, 'BATCH_SIZE =', BATCH_SIZE)

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

train_batches = training.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
validation_batches = validation.padded_batch(BATCH_SIZE)

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
