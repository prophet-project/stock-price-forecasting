import tensorflow as tf
from .libs import params, prepare, save, checkpoints
from .prepare_datasets import get_prepared_datasets
from .model import build_model
from tensorflow.keras.callbacks import CSVLogger
from .window_generator import WindowGenerator

prepare(tf)

LABEL_STEPS = params['train']['label_steps']
MAX_EPOCHS = params['train']['epochs']

metrics_file='metrics/training.csv'

def fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )

    history = model.fit(
        window.train, 
        epochs=MAX_EPOCHS,
        validation_data=window.test,
        callbacks=[
            early_stopping,
            checkpoints.save_weights(), 
            CSVLogger(metrics_file)
        ]
    )

    model.summary()

    return history

def train():
    # Load normalised datasets
    train_df, test_df = get_prepared_datasets()
    # Build neural network model
    model = build_model()

    window = WindowGenerator(
        input_width=30, label_width=LABEL_STEPS, shift=LABEL_STEPS,
        train_df=train_df, test_df=test_df,
        label_columns=['Close']
    )

    fit(model, window)

    # Save for restore in next time
    save(model)

if __name__ == '__main__':
    train()


