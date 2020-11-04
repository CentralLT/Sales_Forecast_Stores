import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
import string


def preprocess(sentence):
    """Preprocessing sentences and applying tokenize. Lowercase words,
    remove stopwords, non ASCI characters and punctuation"""

    sentence = sentence.lower()

    # collect set of stopwords and punctuation
    stopset = stopwords.words('russian') + list(string.punctuation)
    specialset = ['(', ')', '-', '[', ']']
    sentence = ' '.join([i for i in word_tokenize(sentence) if i not in stopset or i not in specialset])
    sentence = unidecode(sentence)

    return sentence
