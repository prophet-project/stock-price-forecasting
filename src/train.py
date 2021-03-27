import tensorflow as tf
from .libs import params, prepare, save, checkpoints
from .prepare_datasets import make_window_generator
from .model import build_model
from tensorflow.keras.callbacks import CSVLogger, Callback

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

    model.fit(
        window.train, 
        epochs=MAX_EPOCHS,
        validation_data=window.test,
        callbacks=[
            checkpoints.save_weights(), 
            CSVLogger(metrics_file),
        ],
    )

    model.summary()


def train():
    # Build neural network model
    model = build_model()

    window = make_window_generator()
    print(window)

    fit(model, window)


    # Save for restore in next time
    save(model)

if __name__ == '__main__':
    train()


