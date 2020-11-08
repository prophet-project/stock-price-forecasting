from pycontractions import Contractions

"""
    Contractions are words that we write with an apostrophe.
    Examples of contractions are words like “ain’t” or “aren’t”.
    For standartize text better replace them
"""

# See description in README.md
cont = Contractions('../data/GoogleNews-vectors-negative300.bin')
cont.load_models()

""" 
    Will expand contractions in list of texts. 
    Better use on big amounts on texts. One by One proecceing will be slow
    expand_texts produce texts in unicode
"""
def replace_contractions_in_text(batch_of_texts):
    return cont.expand_texts(batch_of_texts, precise=True)