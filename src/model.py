from tensorflow.keras import Sequential, layers, losses, optimizers, metrics, initializers
from .libs import params
from .FeedBackModel import FeedBack
from .prepare_datasets import feature_list

BATCH_SIZE=32
NUM_FEATURES=11
INPUT_WIDTH = params['train']['input_width']

INPUT_SHAPE = (BATCH_SIZE, INPUT_WIDTH, NUM_FEATURES)
print(INPUT_SHAPE)

def build_model():
    model = Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        layers.Dense(units=1)
    ])
    
    
    model.compile(
        loss=losses.MeanSquaredError(),
        optimizer=optimizers.Adam(),
        metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredLogarithmicError()]
    )

    return model