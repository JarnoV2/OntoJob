import numpy as np
#import plotly

from sklearn import preprocessing as sk_pp
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.poincare import PoincareModel
from gensim.models.doc2vec import TaggedDocument
from gensim.matutils import corpus2csc
from gensim.corpora import Dictionary
#from gensim.viz.poincare import poincare_2d_visualization

_N_DIMENSION = 200
_N_WINDOW = 5
_N_EPOCHS = 50

def get_w2v(train):
    model = Word2Vec(sentences = train,
            min_count = 1,
            window = _N_WINDOW,
            size = _N_DIMENSION,
            sg = 1,
            hs = 1,
            sample = 1e-5,
            alpha = 0.25,
            min_alpha = 0.0001,
            negative = 4)
    model.train(sentences = train,
            total_examples = model.corpus_count,
            epochs = _N_EPOCHS,
            report_delay = 1)
    return dict(zip(model.wv.index_to_key, sk_pp.normalize(model.wv.vectors, norm = 'l2')))

def get_d2v(train_data):
    docs = LabeledDocuments(train_data)
    model = Doc2Vec(documents = docs,
            vector_size = _N_DIMENSION,
            window = _N_WINDOW,
            min_count = 1,
            workers = 4, 
            dm = 0,
            dbow_words = 1)
    model.train(docs, total_examples = model.corpus_total_words, epochs = _N_EPOCHS)
    return dict(zip(model.wv.index_to_key, model.wv.vectors)), dict(zip(model.docvecs.index_to_key, model.docvecs))

def get_co_occurence_dict(train_data):
    _dct = Dictionary(train_data)
    _bow_corpus = [_dct.doc2bow(line) for line in train_data]
    skill_list = [skill for skill in _dct.values()]
    return np.dot(corpus_csc(_bow_corpus), corpus2csc(_bow_corpus).T)

def get_poincare(train_data):
    model = PoincareModel(train_data, negative = 2)
    model.train(_N_EPOCHS)
    return model 

def get_poincare_visual(model, train_data):
    figure = poincare_2d_visualization(model, tree = set(train_data), figure_title = 'Taxonomy', num_nodes = None, show_node_labels = set(train_data))
    return plotly.offline.plot(figure)

def infer_vector(term_dict, child_list):
    return np.mean([term_dict[child] for child in child_list], axis = 0)

class LabeledDocuments:

    def __init__(self, train_data):
        self.train_data = train_data

    def __iter__(self):
        for k, v in self.train_data.items():
            yield TaggedDocument(words=v, tags=[k])
