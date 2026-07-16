import re
import numpy as np
from nltk.stem.snowball import PorterStemmer

# PorterStemmer uses an algorithm (without downloading data via
# nltk.download), unlike word_tokenize/punkt—so for tokenization,
# we use our own regex instead of nltk.word_tokenize.
stemmer = PorterStemmer() 

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(sentence: str):
    """
    Splits a sentence into words/tokens (Cyrillic and Latin letters, and numbers).
    Punctuation is ignored.
    """
    return _TOKEN_RE.findall(sentence.lower())


def stem(word: str) -> str:
    """
    Finds the root of a word.
    Examples: "learning", "learning" -> "learn"
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    Returns a bag-of-words vector:
    1 if the stemmed word is in the sentence, 0 otherwise.
    """
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
