from .spacy_nlp_model import nlp

# will replace different text forms with one common form
# For example "connected" will be replaced by "connect"

lemmatizer = nlp.vocab.morphology.lemmatizer

# convert work to his root form
# example connected -> connect
def lematize(word):
    return lemmatizer.lookup(word)