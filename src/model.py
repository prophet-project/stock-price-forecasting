from tensorflow.keras import Sequential, layers, losses, optimizers, metrics, initializers
from .libs import params
from .FeedBackModel import FeedBack

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
        metrics=[metrics.MeanAbsoluteError()]
    )

    return model