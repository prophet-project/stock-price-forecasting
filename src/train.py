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
    mode='min',
    verbose=1
)

def fit(model, train, test):

    model.fit(
        train, 
        epochs=MAX_EPOCHS,
        validation_data=test,
        callbacks=[
            checkpoints.save_best_weights(), 
            CSVLogger(metrics_file),
            early_stopping
        ],
    )

    model.summary()


def train():
    # Build neural network model
    model = build_model()

    train, test = make_window_generator()
    fit(model, train, test)
  
    # allow restore model if know structure
    save_weights(model)
    # Save self containing mode, could fail
    save(model)

if __name__ == '__main__':
    train()


