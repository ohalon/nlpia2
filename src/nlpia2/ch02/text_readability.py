from nltk.tokenize import sent_tokenize, word_tokenize
import syllables

def readability_ease_score(text):
    """Return the Flesch readability ease score of a text.
    Interpretation: the more difficult the text is, the lower the score.
    
    >>> readability_ease_score("data science for social good")
    75.88
    >>> readability_ease_score("what time is it?")
    117.16
    >>> readability_ease_score(nan)
    nan
    """
    
    if str(text) != 'nan':
        num_sentences = len(sent_tokenize(text))
        num_words = len(word_tokenize(text))
        num_syllables = sum(syllables.estimate(w) for w in word_tokenize(text))
        score = 206.835 - 1.015 * (float(num_words) / num_sentences) - 84.6 * (num_syllables / float(num_words))
        return round(score, 2)
    else:
        return np.nan
