# Will prepare text for processing


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
special_symbols_re = r'[^a-zA-z0-9\s\!\#\.\'\-\@\/\:]'

def remove_special_characters(text):
    return re.sub(special_symbols_re, '', text)

def remove_hashtag_character(text):
    return text.replace('#', '')

#! need reaplace numbers after we remove hastags
#! and after lematization, for process it like 'Third' -> '3' -> '#'
# Will replace numbers and add spaces, 
# to parse them without errors
def replace_numbers(text, replace_to=' # '):
    if not bool(re.search(r'\d', text)):
        return text
        
    return re.sub('[0-9]{1,}', replace_to, text)
    
def remove_quotes(text):
    return text.replace('"', '').replace("'", '')

