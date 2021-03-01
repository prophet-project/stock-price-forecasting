from tensorflow.keras import Sequential, layers, losses, optimizers, metrics, initializers
from .libs import params
from .FeedBackModel import FeedBack

LABEL_STEPS = params['train']['label_steps']
NUM_FEATURES = params['train']['features']

def build_model():
    model = FeedBack(units=32, out_steps=LABEL_STEPS, num_features=NUM_FEATURES)
    
    model.compile(
        loss=losses.MeanSquaredError(),
        optimizer=optimizers.Adam(),
        metrics=[metrics.MeanAbsoluteError()]
    )

    return model