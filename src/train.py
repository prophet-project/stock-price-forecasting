import tensorflow as tf
from .libs import params, prepare, save, checkpoints
from .prepare_datasets import make_window_generator
from .model import build_model
from tensorflow.keras.callbacks import CSVLogger

prepare(tf)


MAX_EPOCHS = params['train']['epochs']
PATIENCE = params['train']['patience']

metrics_file='metrics/training.csv'

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    mode='min'
)

def fit(model, window):
    history = model.fit(
        window.train, 
        epochs=MAX_EPOCHS,
        validation_data=window.test,
        callbacks=[
            early_stopping,
            checkpoints.save_weights(), 
            CSVLogger(metrics_file)
        ],
        shuffle=False
    )

    model.summary()

    return history

def train():
    # Build neural network model
    model = build_model()

    window = make_window_generator()
    print(window)

    fit(model, window)

    # clear lstm state
    model.reset_states()

    # Save for restore in next time
    save(model)

if __name__ == '__main__':
    train()


