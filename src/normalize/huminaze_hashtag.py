from .split_joined_words import split

""" 
    Will remove hashtag symbol 
    and add spaces if hashtage combined by multiple words
"""
def huminaze_hashtag(word):
    if is_hashtag(word):
        word = word[1:] # remove hashtag symbol

    words = split(word)
    if len(words) == 0:
        return word

    return ' '.join(words)

def is_hashtag(word):
    return word.startswith('#') and len(word) > 1
