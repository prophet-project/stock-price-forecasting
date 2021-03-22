from tensorflow.keras import Sequential, layers, losses, optimizers, metrics, initializers
from .libs import params
from .FeedBackModel import FeedBack
from .prepare_datasets import feature_list, BATCH_SIZE

NUM_FEATURES=8
INPUT_WIDTH = 32

INPUT_SHAPE = (BATCH_SIZE, INPUT_WIDTH, NUM_FEATURES)
print(INPUT_SHAPE)

def build_model():
    model = Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        layers.LSTM(32, return_sequences=True, stateful=True, batch_input_shape=INPUT_SHAPE),
        # Shape => [batch, time, features]
        layers.Dense(units=1)
    ])
    
    
    model.compile(
        loss=losses.MeanSquaredError(),
        optimizer=optimizers.Adam(),
        metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredLogarithmicError()]
    )

    return model