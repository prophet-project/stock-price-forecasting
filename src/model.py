from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import losses

def build_model(input_shape):
    model = Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        # Two dense layer allow make separate predictions about each class
        layers.Dense(2)
    ])

    model.summary()
    model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model