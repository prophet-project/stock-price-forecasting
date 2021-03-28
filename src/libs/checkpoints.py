import os
import tensorflow as tf

# Will save weight during training
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

best_model_path = 'saved_models/best_val_los.h5'

# Create a callback that saves the model's weights every 2 epochs during training
def save_weights():
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=2
    )

# Load saved weights
def load_weights(model):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    return model

def save_best():
    return tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )

def load_best(custom_objects=None):
    new_model = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)

    # Check its architecture
    new_model.summary()
    return new_model

def save_best_weights():
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True, 
        save_weights_only=True
    )