import jamspell
# JamSpell is good library, but it seems out of support.
# It based on language model, 
# instead of dictanory so must get better results then TextBlob and pyspellchecker.
# But in some cases they will be enough

corrector = jamspell.TSpellCorrector()
if corrector.LoadLangModel('./data/en.jamspell.model.bin'):
    print('JamSpell model loaded successfully')
else:
    print('JamSpell model failed to load')

""" 
    Will replace typos and misspells in common words,
    better run after links, usernames and hashtags removed
"""
def replace_misspells(text):
    text = corrector.FixFragment(text)
    return text

""" 
    Check is word correct 
    rating - border when word is correct, number from 0 to 100
        if correct not know word rating less 50, if know then more than 50.
        But you can control this border
"""
def is_correct(word, rating = 50):
    # jamspell not provide api for check is word correct
    # so will make custom version
    candidates = corrector.GetCandidatesWithScores([word], 0)
    first, score = candidates[0] 

    # first candidate must be same word if JamSpell know it
    if first != word:
        return False

    score = abs(score) # score is negative, then less then better
    return score < rating
    


