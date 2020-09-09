import tensorflow as tf
import tensorflow_datasets as tfds
from .datasets import get_train_dataset
from .normalize import encoder_filename, encoder_info_filename
import json

# Need firstly build encoder on training dataset, 
# for embeding of all text input

def build_encoder(labeled_data):
    tokenizer = tfds.features.text.Tokenizer()
    vocabulary_set = set()
    for text_tensor, _ in labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)

    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    vocab_size = len(vocabulary_set)
    return encoder, vocab_size

def save(encoder, info_data):
    encoder.save_to_file(encoder_filename)
    print("Encoder saved to", encoder_filename)

    with open(encoder_info_filename, 'w') as info:
        json.dump(info_data, info)
    
    print('Encoder info saved to', encoder_info_filename)



print('Start building encoder...')

train_data = get_train_dataset(display_progress=True)
encoder, vocab_size = build_encoder(train_data)

print('Encoder was build, saving...')
save(encoder, { 'vocab_size': vocab_size })
