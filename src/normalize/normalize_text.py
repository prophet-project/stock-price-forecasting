from .spacy_nlp_model import nlp
from .clean_text import remove_special_characters, replace_numbers, remove_quotes, remove_sharp_character
from .replace_contractions import replace_contractions
from .remove_stopwords import is_stopword
from .replace_misspells import replace_misspells
from .huminaze_hashtag import is_hashtag, huminaze_hashtag
from .unify_text import is_username, is_link
from .lemmatization import lematize

"""
    Will normalize text by removing specific symbols and stopwords, for decrease noise
    For decrease vocabulary will:
        - unify usernames and links
        - lemmatize words
        - fix misspelling
        - etc.
"""
def normalize_text(text):
    # prepare text for complex parsing
    text = preprocess_text(text) 
    
    # replace misspells after expand hashtag
    # because correcter can change hasthag like `#thebest` to `#thebes`
    # and before replace contractions for replace incorrect contractions
    text = replace_misspells(text)
    # replace contractions for expand to standart words
    # replace_contractions optimized for batches, 
    # but in our case better easy readable code
    text = replace_contractions([text])[0]
    # remove quotes after contractions expandent
    text = remove_quotes(text)

    # Will lematize known words and remove stopwords 
    # for decrease vocabulary size
    text = simplify_text(text)

    # Will replace numbers to '#' for not increase vocabulary size
    # usage it after spacy will replace words with numbers: "Third" -> '3'
    text = replace_numbers(text)

    return text

"""
    Remove simplest noize tokens, like usernames and links
    and specific symbols,
    also expand hashtags as words.
    It allow more intelectual parsers esily updated text
"""
def preprocess_text(text):
    text = remove_special_characters(text)
    
    words = text.split(' ')
    words = [preprocess_word(word) for word in words]
    words = [word for word in words if word is not None]

    text = ' '.join(words)

    # after links and hashtags removed
    # possible some '#' stay in words or numbers
    text = remove_sharp_character(text) 

    return text

def preprocess_word(word):
        
    if len(word) == 0:
        return None

    # will replace usernames and links to one symbol
    # for remove noize specific tokens
    if is_username(word):
        return 'username'

    if is_link(word):
        return 'link'

    # will replace hashtag with humized version
    # for example "#thebest" -> "the best"
    if is_hashtag(word):
        return huminaze_hashtag(word)

    return word

def simplify_text(text):
    # split sentence into tokens (parts)
    doc = nlp(text)

    # Will process each token independently
    # and lematize known words and remove stopwords
    words = [process_token(token) for token in doc]
    words = [word for word in words if word is not None]

    return ' '.join(words)

""" 
    Will process each token and return word 
    or None if token need in result text 
"""
def process_token(token):
    word = token.text
    
    if len(word) == 0:
        # By some reason spacy sometimes return zero length toke
        return None

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