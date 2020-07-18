import os
import tensorflow as tf

# Will save weight during training
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

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