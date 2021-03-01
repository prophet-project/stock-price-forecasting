from tensorflow.keras import Sequential, layers, losses, optimizers, metrics, initializers
from .libs import params

LABEL_STEPS = params['train']['label_steps']
NUM_FEATURES = params['train']['features']

def build_model():
    model = Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features]
        layers.Dense(
            LABEL_STEPS*NUM_FEATURES,
            kernel_initializer=initializers.zeros()
        ),
        # Shape => [batch, out_steps, features]
        layers.Reshape([LABEL_STEPS, NUM_FEATURES])
    ])
    
    model.compile(
        loss=losses.MeanSquaredError(),
        optimizer=optimizers.Adam(),
        metrics=[metrics.MeanAbsoluteError()]
    )

    return model