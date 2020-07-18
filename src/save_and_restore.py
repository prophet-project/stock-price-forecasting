import tensorflow as tf

PATH_TO_SAVED_MODEL='saved_model/main'
PATH_TO_SAVE_HD5_MODEL='saved_model/main.h5'

# Save model in SavedModel format, which can be fully respored
def save(model):
    model.save(PATH_TO_SAVED_MODEL)

def saveHD5(model):
    model.save(PATH_TO_SAVE_HD5_MODEL)

def load():
    new_model = tf.keras.models.load_model(PATH_TO_SAVED_MODEL)

    # Check its architecture
    new_model.summary()
    return new_model

def loadHD5():
    new_model = tf.keras.models.load_model(PATH_TO_SAVE_HD5_MODEL)

    # Check its architecture
    new_model.summary()
    return new_model