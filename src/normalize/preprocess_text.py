from .spacy_nlp_model import nlp
from .clean_text import remove_special_characters, replace_numbers, remove_hashtag_character, remove_quotes
from .replace_contractions import replace_contractions_in_text
from .remove_stopwords import is_stopword
from .replace_misspells import replace_misspells


""" 
    Will make preprocessing, but on texts batch
    Important place batch tasks, like contrctions replacing in that part
"""
def preprocess_batch_of_texts(batch_of_texts):
    batch_of_texts = [decode_text_bytes(text) for text in batch_of_texts]
    batch_of_texts = replace_contractions_in_text(batch_of_texts)

    # TODO: use pool
    for text in batch_of_texts:
        # replace_contractions_in_text produce unicode texts
        text = encode_unicode_text(text) # TODO: check is this step need 
        # remove quotes after contractions expandent
        text = remove_quotes(text)
        # major text processing part
        yield preprocess_text(text)

"""
    Will preprocess text by removing specific symbols and stopwords, for decrease noise
    For decrease vocabulary will:
        - unify usernames and links
        - lemmatize words
        - fix misspelling
"""
def preprocess_text(text):

    text = remove_special_characters(text)
    
    # replace misspells before remove hastag symbol for stay hashtags as it is
    # and before replace contractions for replace incorrect contractions
    text = replace_misspells(text)

    # will stay hashtags as it is, 
    # becuase they can give sentiment information
    # only remove '#' in case they same like words in vocabulary
    text = remove_hashtag_character(text) 

    # split sentence into tokens (parts)
    doc = nlp(text)

    # Will process each token independently
    # and remove common nouns and other special words (usernames, links...)
    # and lematize known words
    words = [preprocess_token(token) for token in doc]
    words = [word for word in words if word is not None]

    text = ' '.join(words)

    # Will replace numbers to '#' for not increase vocabulary size
    # usage it after spacy will replace words with numbers: "Third" -> '3'
    text = replace_numbers(text)

    return text

""" 
    Will process each token and return word 
    or None if token need in result text 
"""
def preprocess_token(token):
    word = token.text
    
    if len(word) == 0:
        # By some reason spacy sometimes return zero length toke
        return None

    # will replace usernames and links to one symbol    
    if is_username(word):
        return '@user'

    if is_link(word):
        return '@link'

    # TODO remove common nouns before lematization
    
    word = word.lower()
    # leematize word for decrease vocabulary size
    word = lematize(word)

    # stopwords is common words like "a", "is", "be", etc.
    # without real meaning they create noize
    if is_stopword(word):
        return None

    return word

""" Tenserflow process text in bytes, so need decode it firstly """
def decode_text_bytes(text_bytes):
    return text_bytes.decode('utf-8')

""" Encode unicode text to standart strings """
def encode_unicode_text(unicode_text):
    return unicode_text.encode('utf-8')