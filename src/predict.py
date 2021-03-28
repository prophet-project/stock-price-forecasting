import tensorflow as tf
import pandas as pd
from .prepare_datasets import make_window_generator
from .libs import params, prepare, save_metrics, load, checkpoints
from .model import build_model

prepare(tf)

predictions_file = './predictions.csv'

# Load normalised datasets
train, test = make_window_generator()

model = build_model()
model = checkpoints.load_weights(model)

predictions = model.predict(
    test, 
    verbose=1,
)

with open(predictions_file, 'w') as outfile:
    df = pd.DataFrame(predictions)
    df.to_csv(outfile)