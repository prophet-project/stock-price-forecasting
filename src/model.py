from tensorflow.keras import Sequential, layers, losses, optimizers, metrics, initializers, backend
from attention import Attention
from .libs import params

def percentage_difference(y_true, y_pred):
    return backend.mean(abs(y_pred/y_true - 1) * 100)

def build_model():
    model = Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        layers.LSTM(128, return_sequences=True),
        Attention(64),
        layers.Dropout(0.2),
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
