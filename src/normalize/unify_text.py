# Allow unfiy text for make specific tokens simular

def is_username(word):
    return word.startswith('@') and len(word) > 1

common_domains = ['.com', '.net', '.org', '.edu', '.io']
minimal_url_length = 5 # url like go.io is mostly shrter url

def is_link(word):
    if len(word) < minimal_url_length:
        return False

    if word.startswith('http') or word.startswith('www.'):
        return True

    found_domains = [domain for domain in common_domains if domain in word]
    if len(found_domains) > 0:
        return True

    return False

# Can be repeat multiple times in text
repeatable_punctuation_symbols = ['!', '_', '-', '=', '+']

# replace symbols which repeats much times
def minimize_repetative_symbols(text):
    # will skip firts symbol
    result = text[:1]

    for symbol in repeatable_punctuation_symbols:
        for i in range(1, len(text)):
            
            if text[i] == symbol and text[i - 1] == symbol:
                # if current and previus symbols same, then not add it
                continue

            result = result + text[i]

    return result

def findAll(str, substring):
    return [m.start() for m in re.finditer(substring, str)]

def reaplace_by_indexes(str, indexes, replacement):
    result = []
    for i in indexes:
        result.append(str[:index])      
        result.append(replacement)
        result.append(str[index:])  
    
    return "".join(result)

# When someone write 'this is citate' tokenizer can not correctly parse it
# will add space before and after it
# but we cannot because contractions in engslish language
contractions_endings = ["'t", "'d","'ll","'m","'re","'s","'ve","'d","'ll","'m","'re","'s","'ve"]
# so will escape them while parsing

""" Will replace all accouriences for ' (quote), except for shortcuts """
def replace_quotes(text, replace_by=''):
    shortcut_indexes = []
    for shortcut in contractions_endings:
        found = findAll(text, shortcut)
        shortcut_indexes.extend(found)

    quotes_found = findAll(text, "'")
    need_replace = [i for i in quotes_found if i not in shortcut_indexes]

    return reaplace_by_indexes(text, need_replace, replace_by)

""" Will add space after and before ' (quote), except for shortcuts """
def dress_quotes(text):
    return reaplace_by_indexes(text, replace_by=" ' ")