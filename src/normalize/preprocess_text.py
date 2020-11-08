from .spacy_nlp_model import nlp
from .clean_text import remove_special_characters, replace_numbers, remove_hashtag_character, remove_quotes
from .replace_contractions import replace_contractions
from .remove_stopwords import is_stopword

"""
    Will preprocess text by removing specific symbols and stopwords, for decrease noise
    For decrease vocabulary will:
        - unify usernames and links
        - lemmatize words
        - fix misspelling
"""
def preprocess_text(text_bytes):
    text = text_bytes.decode('utf-8')

    text = remove_special_characters(text)
    
    # will stay hashtags as it is, 
    # becuase they can give sentiment information
    # only remove '#' in case they same like words in vocabulary
    text = remove_hashtag_character(text) 

    text = replace_contractions(text)
    text = remove_quotes(text)

    # split sentence into tokens (parts)
    doc = nlp(text)

    text = ' '.join(preprocess_tokens(doc))

    # Will replace numbers to '#' for not increase vocabulary size
    # usage it after spacy will replace words with numbers: "Third" -> '3'
    text = replace_numbers(text)

    return text

def preprocess_tokens(doc):
    for token in doc:
        word = token.text
        
        if len(word) == 0:
            # By some reason spacy sometimes return zero length toke
            continue

        # will replace usernames and links to one symbol    
        if is_username(word):
            yield '@user'
            continue

        if is_link(word):
            yield '@link'
            continue

        # leematize word for decrease vocabulary size
        word = lematize(word.lower())

        # stopwords is common words like "a", "is", "be", etc.
        # without real meaning they create noize
        if is_stopword(word):
            continue

        yield word