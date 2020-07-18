from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import losses

def build_model(input_shape):
    model = Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])

    model.summary()
    model.compile(
        optimizer='adam',
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model