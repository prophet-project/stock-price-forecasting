from tensorflow.keras import Sequential, layers, losses, optimizers, metrics, initializers, backend
from .libs import params
from .FeedBackModel import FeedBack
from .prepare_datasets import feature_list

def percentage_difference(y_true, y_pred):
    return backend.mean(abs(y_pred/y_true - 1) * 100)

def build_model():
    model = Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        layers.Dense(units=1)
    ])
    
    compile(model)
    
    return model

def compile(model):
    model.compile(
        loss=losses.MeanSquaredError(),
        optimizer=optimizers.Adam(),
        metrics=[
            metrics.MeanAbsoluteError(),
            metrics.MeanSquaredLogarithmicError(),
            percentage_difference
        ]
    )
