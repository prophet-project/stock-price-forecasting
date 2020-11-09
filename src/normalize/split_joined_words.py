from .replace_misspells import is_correct

""" 
    can be usefull for hastags parsing, like #wontfollow.
    copy of https://github.com/TimKam/compound-word-splitter/blob/master/splitter/compound_word_splitter.py
    but with usage of jamspell instead of enchant
"""
def split(word):
    max_index = len(word)

    for index, char in enumerate(word):
        
        left_compound = word[0:max_index-index]
        right_compound_1 = word[max_index-index:max_index]
        right_compound_2 = word[max_index-index+1:max_index]

        is_left_compound_valid_word = len(left_compound) > 1 and is_correct(left_compound)

        if is_left_compound_valid_word and \
                (not split(right_compound_1) == '' or right_compound_1 == ''):
            return [compound for compound in concat(left_compound, split(right_compound_1))\
                    if not compound == '']
        
        elif is_left_compound_valid_word and word[max_index-index:max_index-index+1] == 's' and \
            (not split(right_compound_2) == '' or right_compound_2 == ''):
            return [compound for compound in concat(left_compound, split(right_compound_2))\
                    if not compound == '']
    
    if not word == '' and is_correct(word):
        return [word]
    else:
        return ''

def concat(object1, object2):
    if isinstance(object1, str):
        object1 = [object1]
    if isinstance(object2, str):
        object2 = [object2]
    return object1 + object2
