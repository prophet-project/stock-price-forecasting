import tensorflow as tf
import numpy as np
import datasets
from normalize import normalize_datasets
from model import build_model
from train import train
from test import test


print("Tensorflow:", tf.__version__)

# Load dataset
train_data, validation_data, test_data = datasets.download()
# Dataset data is array of tensors
# if symplify array of tuples: (text: string, label: int)
# where 0 mean bad, and 1 mean good

# Normalise data
training, validation, testing, input_shape = normalize_datasets(train_data, validation_data, test_data)

# Build neural network model
model = build_model(input_shape)

# Train network
train(model, training, validation)

# Test network
test(model, testing)