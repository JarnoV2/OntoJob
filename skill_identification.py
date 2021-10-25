import gensim
import numpy as np
import spacy
import nltk

from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

from gensim import corpora
from gensim.utils import simple_preprocess
from collections import defaultdict, OrderedDict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer

NLP = spacy.load('en_core_web_lg')
ENT_LIST = ['LOC', 'GPE'] #['PRODUCT', 'ORG', 'EVENT', 'LAW', 'LANGUAGE']


def get_bigrams(s):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(s)
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)
    [tokens.append("%s %s".replace(' ', '_') % bigram) for bigram in bigrams]
    return ["".join(token) for token in tokens]

def get_trigrams(s):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(s)
    trigram_finder = TrigramCollocationFinder.from_words(tokens)
    trigrams = trigram_finder.nbest(TrigramAssocMeasures.pmi, 10)
    [tokens.append("%s %s %s".replace(' ', '_') %trigram) for trigram in trigrams]
    return ["".join(token) for token in tokens]

def get_ngrams(s):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(s)
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    trigram_finder = TrigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)
    trigrams = trigram_finder.nbest(TrigramAssocMeasures.pmi, 10)
    [tokens.append("%s %s".replace(' ', '_') % bigram) for bigram in bigrams]
    [tokens.append("%s %s %s".replace(' ', '_') % trigram) for trigram in trigrams]
    return ["".join(token) for token in tokens]

def get_top_k(corpus, n):
    out = defaultdict(dict)
    result = defaultdict(list)
    dictionary = corpora.Dictionary()
    tokenized_docs = [simple_preprocess(doc) for doc in corpus]
    pvdbow = [dictionary.doc2bow(doc, allow_update = True) for doc in tokenized_docs]
    tfidf = gensim.models.TfidfModel(pvdbow, smartirs = 'ntc')
    [[out[count].update({dictionary[id] : freq}) for id, freq in doc] for count, doc in enumerate(tfidf[pvdbow])]
    [[result[count].append(k) for k, v in Counter(out[count]).most_common(n)] for count in out.keys()] 
    return result 

#def get_top_n(corpus, n):
#    out = defaultdict(list)
#    vectorizer = TfidfVectorizer()
#    X = vectorizer.fit_transform(corpus)
#    labels = np.array(vectorizer.get_feature_names())
#    [out[i].append(labels[np.argsort(X[i].toarray()).flatten()[::-1]][:n]) for i in range(0, X.shape[0])]
#    return out

if __name__ == "__main__":
    print('running')
