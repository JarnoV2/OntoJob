import pandas as pd
import numpy as np
import logging
import re
import multiprocessing as mp

from pattern_learning import discover_patterns, extract_patterns, validate_patterns, validate_multi_terms, validate_h_patterns

from collections import defaultdict, Counter
from itertools import chain, combinations
from tqdm import tqdm

from preprocessing import load_data, load_unprocessed_data, process_string, PRE_EMBEDDING_FILTERS 

from pretrain import get_clustering
from utils import get_intersection, get_vectors, rbtw, invert_dict, drop_low_occurrences, merge_dict, chunks 
from matutils import predict_relation 
from ontology import onto

logging.basicConfig(filename='application.log', 
                    level=logging.DEBUG, 
                    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                    )

MIN_DF = 0.05
MAX_DF = 0.80
CROP_MARGIN = 0.20
SEED_LIST = ['NOUNcommunication', 'NOUNsales', 'NOUNteamwork', 'NOUNplanning', 'NOUNwriting', 'NOUNprogramming', 'NOUNmanaging', 'NOUNcollaboration', 'NOUNpython', 'NOUNresearch',
             'NOUNoracle', 'NOUNspanish', 'NOUNjava', 'NOUNteaching', 'NOUNcleaning', 'NOUNtroubleshooting', 'NOUNenglish', 'NOUNnet', 'NOUNaccounting', 'NOUNsql']


#['NOUNpython', 'NOUNteamwork', 'NOUNcommunication', 'NOUNmanagement', 'NOUNdevelopment', 'NOUNwriting', 'NOUNpresentation', 
            # 'NOUNjava', 'NOUNperl', 'NOUNsql', 'NOUNruby', 'NOUNaccounting', 'NOUNcollaboration', 'NOUNnegotiation', 'NOUNorganization',
            # 'NOUNsales', 'NOUNprogramming', 'NOUNnetworking', 'NOUNc', 'NOUNnet', 'NOUNmatlab', 'NOUNart', 'NOUNlinux', 'NOUNwindows'] 

def get_patterns(raw_df):
    pattern_list, val_dict = discover_patterns(raw_df, SEED_LIST)
    pattern_list = validate_patterns(val_dict, pattern_list, SEED_LIST)
    logging.info('{} DISCOVERED PATTERNS after validation'.format(len(pattern_list)))
    single_term_dict, multi_term_val_dict, top_n_multi_term_qualities_list, hierarch_relation_dict, context_pattern_list = extract_patterns(raw_df, pattern_list, extract_context_words=False)
    # VALIDATE THE TAX RELS here
    logging.debug('TOP_QUALITIES: {}'.format(top_n_multi_term_qualities_list))
    logging.debug('PATTERN_LIST: {}'.format(pattern_list))
    logging.debug('SINGLE_TERM_DICT: {}'.format(single_term_dict))
    logging.debug('CONTEXT_PATTERN_LIST: {}'.format(context_pattern_list))
    logging.debug('HIERARCH_RELATION_DICT: {}'.format(hierarch_relation_dict.values()))
    #for pattern in context_pattern_list:
    #    if pattern[1] == 'first':
    #        if pattern[0].split(' ')[0] in top_n_multi_term_qualities_list:
    #            continue
    #        else:
    #            context_pattern_list.remove(pattern)
    #    else:
    #        if pattern[0].split(' ')[-1] in top_n_multi_term_qualities_list:
    #            continue
    #        else:
    #            context_pattern_list.remove(pattern)
    #context_word_dict = extract_patterns(raw_df, context_pattern_list, extract_context_words=True)
    for k, v in multi_term_val_dict.items():
        multi_term_val_dict.update({k:[e for e in v if e in top_n_multi_term_qualities_list]})
    logging.debug('MULTI_TERM_VAL_DICT: {}'.format(multi_term_val_dict))
    #multi_term_dict = validate_multi_terms(multi_term_val_dict, context_word_dict, c_value=False) 
    #logging.debug('MULTI_TERM_DICT: {}'.format(multi_term_dict))
    wq_dict = merge_dict(single_term_dict, multi_term_val_dict)
    logging.debug('WQ_DICT AFTER MERGE SINGLE AND MULTI TERM: {}'.format(wq_dict))
    return wq_dict, hierarch_relation_dict

def extract_worker_qualities(path, job_title_col, job_description_col):
    raw_df = load_unprocessed_data(path, job_title_col, job_description_col)
    wq_dict, hierarch_relation_dict = get_patterns(raw_df)
    logging.info('WQ_DICT: {}'.format(wq_dict))
    for k, v in wq_dict.items():
        try:
            wq_dict[k] = [process_string(e, filters=PRE_EMBEDDING_FILTERS) for e in v]
        except Exception:
            logging.exception('Exception for wq_dict item {} with value {}'.format(k, v))
            continue
    word_vec_dict, doc_vec_dict = get_d2v(wq_dict)
    return  raw_df, wq_dict, word_vec_dict, doc_vec_dict, get_clustering(word_vec_dict, 5)

def get_worker_quality_relations(vocab, h_rel_dict):
    wq_dict = defaultdict(dict)
    exclusion_list = h_rel_dict['NOUNbenefits'] + h_rel_dict['NOUNrequirements']
    vocab = [e for e in vocab if e not in exclusion_list]
    print('Getting worker quality relations...')
    for k, v in tqdm(df.items()):
        wq_dict[k] = Counter(re.findall(r"({})".format("|".join(vocab)), v))
    return wq_dict

def construct_ontology(wq_dict):
    for count, (k, v) in enumerate(wq_dict.items()):
        tmp_ls = [onto.WorkerQuality(e) for e in v]
        tmp = onto.JobTitle(k)
        tmp.requires_worker_qualities.extend(tmp_ls)
    #return onto

def main():
    return NotImplementedError

if __name__ == '__main__':
    df = load_unprocessed_data('data/input_ICSC.csv', 'job_title', 'job_description')
    pd.DataFrame.from_dict(df, orient='index').to_csv("data/clean_df.csv")
    wq_dict, hierarch_relation_dict = get_patterns(df)
    vocab = [*np.unique([*chain.from_iterable([v for v in wq_dict.values()])])]
    h_rels = validate_h_patterns(hierarch_relation_dict)
    h_rels_dict = defaultdict(list)
    pred_dict = predict_relation(h_rels)
    h_rels = [k for k, v in pred_dict.items() if v > 1]
    for i in h_rels:
        try:
            tmp = i.split(',')
            h_rels_dict[tmp[0]].append(tmp[1])
        except:
            h_rels_dict[i[0]].append(i[1])
    # ADD validation for found relations after prediction, since we need to exclude relations p(y, x) < p(x, y)
    # wq_dict = get_worker_quality_relations(vocab, h_rels_dict)
    # construct_ontology(wq_dict)
