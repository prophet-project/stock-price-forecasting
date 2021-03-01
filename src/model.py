from tensorflow.keras import Sequential, layers, losses, optimizers, metrics

def build_model():
    model = Sequential([
        layers.Dense(units=1)
    ])
    
    model.compile(
        loss=losses.MeanSquaredError(),
        optimizer=optimizers.Adam(),
        metrics=[metrics.MeanAbsoluteError()]
    )

    return model