
BUFFER_SIZE=500 # Must be grater or equal to batches size
BATCHES=256 # Allow parallel training, but bigger batch may overfit

def train(model, train_data, validation_data):
    model.fit(
        train_data.shuffle(BUFFER_SIZE).batch(BATCHES),
        epochs=10,
        validation_data=validation_data.batch(BATCHES)
    )