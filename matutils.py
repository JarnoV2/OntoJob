import numpy as np
import scipy.sparse as sparse
import math

from collections import defaultdict, Counter
from itertools import chain, permutations

def get_p(relation_list):
    relation_probability_dict = defaultdict(float)
    n_relations_dict = Counter(relation_list)
    N = sum(n_relations_dict.values())
    for rel in relation_list:
        relation_probability_dict.update({rel:n_relations_dict[rel] / N})
    return relation_probability_dict

def get_n_rel_type_dict(relation_list, i):
    n_dict = defaultdict()
    n_relations = Counter([v.split('~')[i] for v in relation_list])
    N = sum(n_relations.values())
    for k in n_relations.keys():
        n_dict.update({k: n_relations[k] / N}) 
    return  n_dict

def get_ppmi(L, R, relation_probability_dict, p_hypon, p_hyper):
    try:
        return max(0, math.log(relation_probability_dict['{}~{}'.format(L, R)] / (p_hypon[L] * p_hyper[R])))
    except:
        return 0


def get_PPMI(relation_probability_dict, relation_list, ordered_terms):
    hypon_dict = get_n_rel_type_dict(relation_list, 0)
    hyper_dict = get_n_rel_type_dict(relation_list, 1)
    M = sparse.dok_matrix((len(ordered_terms), len(ordered_terms)))
    for count_L, L in enumerate(ordered_terms):
        for count_R, R in enumerate(ordered_terms):
            if L == R:
                M[count_L, count_R] = 0
            else:
                M[count_L, count_R] = get_ppmi(L, R, relation_probability_dict, hypon_dict, hyper_dict)
    return M   

def get_spmi(M):
    U, S, V = sparse.linalg.svds(M)
    return U.dot(np.diag(S)), V.T

def predict(L, R, U, V):
    return U[L].dot(V[R])

def predict_relation(relation_list):
    out = defaultdict(float)
    relation_probability_dict = get_p(relation_list)
    ordered_terms = [*np.unique([k.split('~') for k in relation_probability_dict.keys()])]
    M = get_PPMI(relation_probability_dict, relation_list, ordered_terms)
    U, V = get_spmi(M)
    S = [*permutations(ordered_terms, 2)]
    for s in S:
        out[s] = predict(ordered_terms.index(s[0]), ordered_terms.index(s[1]), U, V)    
    return out   

