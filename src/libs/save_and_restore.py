import tensorflow as tf

PATH_TO_SAVED_MODEL='saved_models/main.h5'
PATH_TO_SAVED_CUSTOM_MODEL='saved_models/main'

# Save model in H5 format, which can be fully respored
def save(model):
    model.save(PATH_TO_SAVED_MODEL)

def load(custom_objects=None):
    new_model = tf.keras.models.load_model(PATH_TO_SAVED_MODEL, custom_objects=custom_objects)

    # Check its architecture
    new_model.summary()
    return new_model

# If model is custom class default save methods will be not enough
# use this methods

def save_weights(model):
    model.save_weights(PATH_TO_SAVED_CUSTOM_MODEL)

# Require model class for restore
def load_weights(model):
    loaded = model.load_weights(PATH_TO_SAVED_CUSTOM_MODEL)

    # loaded.assert_consumed()

    return model