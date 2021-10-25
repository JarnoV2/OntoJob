import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from itertools import chain, islice 
from functools import reduce
from numpy.linalg import norm

_N_SIMULATIONS = 100
MIN_FREQ = 200 
MAX_FREQ = 3500 


def get_intersection(list_1, list_2):
    return list(set(list_1).intersection(set(list_2)))

def btw(a, c, b):
    return norm(a.dot(b)) / (norm(a.dot(c)) + norm(b.dot(c)))

def rbtw(P, R, Q):
    return btw(np.amax(P, 0), np.mean(R, 0), np.amax(Q, 0)) 

def get_pairwise_cosine_similarity(a, b):
    return cosine_similarity(a, b)

def get_vectors(term_list, vector_dict):
    return [vector_dict[term] for term in term_list if term in vector_dict.keys()]

def get_relation_list(rel_dict, direction = ['lower', 'upper']):
    if direction == 'lower':
        return [[[k, v] for v in values]for k, values in rel_dict.items()]
    if direction == 'upper':
        return [[[v, k] for v in values] for k, values in rel_dict.items()]

def invert_dict(term_dict):
    out = defaultdict(list)
    [[out[e].append(k) for e in v] for k, v in term_dict.items()]
    return out 

def get_term_freq_dict(wq_dict):
    return Counter(list(chain.from_iterable([np.unique([e for e in v]).tolist() for v in wq_dict.values()])))

def update_in_place(a, b):
    a.update(b)
    return a

def drop_low_occurrences(wq_dict, min_df, max_df):
    min_df = int(min_df * len(wq_dict.keys()))
    max_df = int(max_df * len(wq_dict.keys()))
    term_dict = get_term_freq_dict(wq_dict)
    stack = []
#    drop_list = np.unique([k for k, v in term_dict.items() if in_interval(v, min_df, max_df) == False]).tolist()
    drop_list = [*set([k if in_interval(v, min_df, max_df) == False else stack.append(k) for k, v in term_dict.items()])]
    total_freq_terms = reduce(update_in_place, (Counter(wq_dict[k]) for k in stack))
    [drop_list.append(k) for k, v in total_freq_terms.items() if in_interval(v, MIN_FREQ, MAX_FREQ) == False]
    drop_list = dict.fromkeys(drop_list, 1)
    for k, v in wq_dict.items():
        wq_dict[k] = [e for e in v if drop_list.get(e) == None]
    return wq_dict, drop_list    

def in_interval(v, min_v, max_v):
    if min_v <= v <= max_v:
        return True
    else:
        return False

def get_adjacency_dict():
    adj_dict = defaultdict(list)
    return NotImplementedError

def get_adjacency_matrix(adjacency_dict, skill_list):
    return np.array([adjacency_dict[skill][0] for skill in skill_list])

def get_local_clustering_coefficient():
    return NotImplementedError

def power_iteration(adjacency_matrix):
    b_k = np.random.rand(adjacency_matrix.shape[1])
    for _ in range(_N_SIMULATIONS):
        b_k1 = np.dot(adjacency_matrix, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    return b_k    

def get_eigenvector_centrality(adjacency_matrix, skill_list):
    eig_dict = defaultdict(list)
    eig_vec = power_iteration(adjacency_matrix)
    [eig_dict[list(skill_list)[count]].append(eig_vec[count]) for count, node in enumerate(adjacency_matrix)]
    return NotImplementedError

def get_transversality_index():
    return NotImplementedError

def merge_dict(d1, d2):
    for k, v in d1.items():
        d2[k].extend(v)
    return d2 

def sort_dict_by_values(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}  

def intersect_dict(d1, d2):
    out = defaultdict(list)
    for k, v in d1.items():
        for e in v:
            if e in d2[k]:
                out[k].append(e)
    return out

def is_nested_list(ls):
    return any(isinstance(i, list) for i in ls)

def chunks(data, batch_size=1000):
    it = iter(data)
    out = []
    for i in range(0, len(data), batch_size):
        out.append(islice(it, batch_size))
    return out

#### EVERYTHING BELOW IS WIP ####
### Temp documentation pertaining to the usage ---- for building the measures.

#def precision(c_l, c_r, measure = None):
#    """
#    Precision measure usable for both the lexical as the taxonomic layer.
#
#    """
#
#    if measure == None:
#        c_l = set(c_l)
#        c_r = set(c_r)
#        return abs(c_l.intersect(c_r)) / abs(c_l)
#    else:
#        return = abs(measure(c_l, c_r)) / abs(measure(c_l, c_r))

def recall(c_l, c_r, measure = None):
    return precision(c_r, c_r, measure)

def tax_precision(c, wq_dict_l, wq_dict_r, sub_d, sup_d, measure):
    inter_ls = wq_dict_l.keys() & wq_dict_r.keys()
    
    return NotImplementedError

def csc(inter_ls, sub_d, sup_d):
    """
    Common Semantic Cotopy measure to get the parents and child of every element e in inter_ls
    """
    out = defaultdict(list)
    for c in inter_ls:
        stack = []
        if sub_d.get_key(c) != None:
            # CHECK if the ordering in the hearst dictionaries is working
            # O(n^3)
            out[c].append([*set(sub_d[c])])
            stack.append(sub_d[c])
        if sup_d.get_key(c) != None:
            out[c].insert([*set(sup_d[c])])
            stack.append(sup_d[c])
    return NotImplementedError

def get_n_rels(c, rel_dict):
    """
    Method to list the number of semantic cotopy's of a certain concept c.
    """
    return [v for k, v in rel_dict if k == c]
