from tensorflow.keras import Sequential, layers, losses, optimizers, metrics, initializers
from .libs import params
from .FeedBackModel import FeedBack

def build_model():
    model = Sequential(
        layers.Dense(units=1)
    )
    
    model.compile(
        loss=losses.MeanSquaredError(),
        optimizer=optimizers.Adam(),
        metrics=[metrics.MeanAbsoluteError()]
    )

    return model