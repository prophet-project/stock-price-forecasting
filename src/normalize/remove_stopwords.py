from .spacy_nlp_model import nlp

"""
    Allow remove stopwords from text
    For decrease size of sentencies
"""


# Will use library stopwords list,
# but remove some words which can be usefull for sentiment analyse
stopwords_exceptions_list = { 
    "'d", 'again', 'against', 'any', 'but', 'can', 'cannot',
    'if', 'else', 'every', 'go', 'i', 'he', 'her', 'his', 'him',
    'it', 'less', 'might', 'me', 'may', 'must', 'more', 'my', "n't",
    'never', 'not', 'no', 'once', 'our', 'should', 'could', 'such',
    'unleast', 'upseat', 'we', 'when', 'whatever', 'who', 'why', 
    'what', 'you', 'yet', 'yours', 'yourself', 'yourselves', 'would',
    'with', 'within', 'without', 'at', 'all', 'it', 'here', 'see', 
    'all', 'over', 'there', 'only', 'how',' did', 'they', 'was', 
    'got', 'miss', 'and', 'about', 'still', 'up', 'this', 'too', 'much',
    'nothing', 'where', 'everyone', 'very', 'down', 'last', 'back',
    'anyone', 'so', 'already', 'empty', 'in', 'better', 'almost', 'out',
    'off', 'will', 'skip', 'now', 'take', 'around', 'another', 'someone',
    'move', 'enough', 'together', 'everithing', 'take', 'make',
    'yes', 'no', 'out', 'get', 'from', 'under'
}
stopwords_list = set(word for word in nlp.Defaults.stop_words if word not in stopwords_exceptions_list)

def is_stopword(word):
    return word in stopwords_list