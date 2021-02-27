import tensorflow as tf

PATH_TO_SAVED_MODEL='saved_models/main.h5'

# Save model in H5 format, which can be fully respored
def save(model):
    model.save(PATH_TO_SAVED_MODEL)

def load():
    new_model = tf.keras.models.load_model(PATH_TO_SAVED_MODEL)

    # Check its architecture
    new_model.summary()
    return new_model