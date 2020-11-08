from textblob import TextBlob

""" 
    Will replace typos and misspells in common words,
    better run after links, usernames and hashtags removed
"""
def replace_misspells(text):
    text = replace_misspells_by_text_blob(text)
    # Possible also use pyspellchecker, but it have limited vocabulary
    # in some cases it can be usefull if find bigger word frequence list
    # https://github.com/barrust/pyspellchecker

    # also good library is JamSpell, but it seems out of support.
    # It based on language model, instead of dictanory
    # And can be good in another cases
    # https://github.com/bakwc/JamSpell
    return text

""" 
    Will replace typos and misspells in common words,
    dictianory based method 
"""
def replace_misspells_by_text_blob(text):
    blob = TextBlob(text)
    return str(blob.correct())



def split_words_without_spaces(text):
    # TODO  
    # can be usefull for hastags parsing, like #wontfollow.
    # one and two spaces will be enough,
    # most hastags maximum two words and some with `the`
    return text