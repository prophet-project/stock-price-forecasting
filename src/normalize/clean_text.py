import re

"""
 Will prepare text for processing
"""

# In that case need use only regex, 
# because symbols can be and unicode symbols, 
# which you cannot expect. And any list of symbols
# will be not enough
# This regex will remove all except 
# letters, numbers, space, 
# punctiations symbols (! . ' -), 
# symbol for identify hashtag (#), 
# symbol for identify email (@),
# symbols for identify links (/ :),
special_symbols_re = r'[^a-zA-z0-9\s\!\#\.\'\-\@\/\:]|[\_\\]'

def remove_special_characters(text):
    return re.sub(special_symbols_re, '', text)

def remove_sharp_character(text):
    return text.replace('#', '')

def remove_repetative_spaces(text):
    return re.sub(' {2,}', ' ', text)

#! need replace numbers after we remove hastags
#! and after lematization, for process it like 'Third' -> '3' -> '#'
# Will replace numbers and add spaces, 
# for parse them without errors
# will stay number size, for better processing
def replace_numbers(text, ):
    if not bool(re.search(r'\d', text)):
        return text
        
    text = re.sub('[0-9]{5,}', ' ##### ', text)
    text = re.sub('[0-9]{4}', ' #### ', text)
    text = re.sub('[0-9]{3}', ' ### ', text)
    text = re.sub('[0-9]{2}', ' ## ', text)
    text = re.sub('[0-9]{1}', ' # ', text)

    text = remove_repetative_spaces(text)

    return text
    
def remove_quotes(text):
    return text.replace('"', '').replace("'", '')

