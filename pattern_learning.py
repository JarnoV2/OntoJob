import numpy as np
from numpy.linalg import svd
import re
import math
import logging

from collections import defaultdict, Counter
from itertools import chain
from heapq import nlargest
from tqdm import tqdm

from utils import merge_dict, sort_dict_by_values, is_nested_list, invert_dict
from matutils import get_p

H_PATTERN_VAL = defaultdict(list)
PATTERN_LIST = [[r'((NOUN\w+ ){2,5})', 'full'],                                              # 
                [r'(((((ADJ\w+)|(NOUN\w+)) ){1,5})(NOUN\w+))', 'full'],
                [r'((ADJ\w+ |NOUN\w+ )+|((ADJ\w+ |NOUN\w+ )*(NOUN\w+ PREP\w+ )?)(ADJ\w+ |NOUN\w+ )*)NOUN\w+', 'rfull']]
                
HEARST_PATTERN = [[r'(NOUN\w+ (, )?ADJsuch PREPas (NOUN\w+ ?(, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(ADJsuch NOUN\w+ (, )?PREPas (NOUN\w+ ?(, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?ADJother NOUN\w+)', 'hlast'],
                  [r'(NOUN\w+ (, )?VERBinclude (NOUN\w+ ?(, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(NOUN\w+ (, )?ADVespecially (NOUN\w+ ?(, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?DETany ADJother NOUN\w+)', 'hlast'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?DETsome ADJother NOUN\w+)', 'hlast'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?VERBbe DETa NOUN\w+)', 'hlast'],
                  [r'(NOUN\w+ (, )?PREPlike (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'ADJsuch (NOUN\w+ (, )?PREPas (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?PREPlike ADJother NOUN\w+)', 'hlast'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?NUMone PREPof DETthe NOUN\w+)', 'hlast'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?NUMone PREPof DETthese NOUN\w+)', 'hlast'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?NUMone PREPof DETthose NOUN\w+)', 'hlast'],
                  [r'NOUNexample PREPof (NOUN\w+ (, )?VERBbe (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?VERBbe NOUNexample PREPof NOUN\w+)', 'hlast'],
                  [r'(NOUN\w+ (, )?PREPfor NOUNexample (, )? (NOUN\w+ ?(, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?DETwhich VERBbe NOUNcall NOUN\w+)', 'hlast'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?DETwhich VERBbe NOUNname NOUN\w+)', 'hlast'],
                  [r'(NOUN\w+ (, )?ADVmainly (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(NOUN\w+ (, )?ADVmostly (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(NOUN\w+ (, )?ADVnotably (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(NOUN\w+ (, )?ADVparticularly (NOUN\w+ ?(, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(NOUN\w+ (, )?ADVprincipally (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(NOUN\w+ (, )?PREPin ADJparticular (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(NOUN\w+ (, )?PREPexcept (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(NOUN\w+ (, )?ADJother PREPthan (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(NOUN\\w+ (, )?NOUNe.g. (, )?(NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(NOUN\w+ \\( (NOUNe.g.|NOUNi.e.) (, )?(NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+(\\. )?\\))', 'hfirst'],
                  [r'(NOUN\w+ (, )?NOUNi.e. (, )?(NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand|CCONJor)? DETa NOUNkind PREPof NOUN\w+)', 'hlast'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand|CCONJor)? NOUNkind PREPof NOUN\w+)', 'hlast'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand|CCONJor)? NOUNform PREPof NOUN\w+)', 'hlast'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?DETwhich VERBlook PREPlike NOUN\w+)', 'hlast'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?DETwhich NOUNsound PREPlike NOUN\w+)', 'hlast'],
                  [r'(NOUN\w+ (, )?DETwhich VERBbe ADJsimilar PRPto (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(NOUN\w+ (, )?NOUNexample PREPof DETthis VERBbe (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(NOUN\w+ (, )?NOUNtype (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )? NOUN\w+ NOUNtype)', ''],
                  [r'(NOUN\w+ (, )?PREPwhether (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(VERBcompare (NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?PREPwith NOUN\w+)', 'hlast'],
                  [r'(NOUN\w+ (, )?VERBcompare PRPto (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'(NOUN\w+ (, )?PREPamong -PRON- (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+)', 'hfirst'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand |CCONJor )?PREPas NOUN\w+)', 'hlast'],
                  [r'(NOUN\w+ (, )? (NOUN\w+ ? (, )?(CCONJand |CCONJor )?)+ PREPfor NOUNinstance)', 'hfirst'],
                  [r'((NOUN\w+ ?(, )?)+(CCONJand|CCONJor)? NOUNsort PREPof NOUN\w+)', 'hlast'],
                  [r'(NOUN\w+ (, )?DETwhich VERBmay VERBinclude (NOUN\w+ ?(, )?(CCONJand |CCONJor )?)+)', 'hfirst']]

def discover_patterns(df, seed_instances):
    patterns= []
    val_dict = defaultdict(list)
    for v in df.values():
        tmp_v = v.split(' ')
        for count, word in enumerate(tmp_v):
            if word in seed_instances:
                try:
                    patterns.append([r'{} {} {} {} {}'.format(tmp_v[count-4], tmp_v[count-3], tmp_v[count-2], tmp_v[count-1], r'\bNOUN\w+'), 'last'])
                    val_dict[patterns[-1][0]].append(word)
                    patterns.append([r'{} {} {} {}'.format(tmp_v[count-3], tmp_v[count-2], tmp_v[count-1], r'\bNOUN\w+' ),'last'])
                    val_dict[patterns[-1][0]].append(word)
                    patterns.append([r'{} {} {}'.format(tmp_v[count-2], tmp_v[count-1], r'\bNOUN\w+'), 'last'])
                    val_dict[patterns[-1][0]].append(word)
                    patterns.append([r'{} {} {}'.format(r'\bNOUN\w+', tmp_v[count+1], tmp_v[count+2]), 'first'])
                    val_dict[patterns[-1][0]].append(word)
                    patterns.append([r'{} {} {} {}'.format(r'\bNOUN\w+', tmp_v[count+1], tmp_v[count+2], tmp_v[count+3]), 'first'])
                    val_dict[patterns[-1][0]].append(word)
                    patterns.append([r'{} {} {} {} {}'.format(r'\bNOUN\w+', tmp_v[count+1], tmp_v[count+2], tmp_v[count+3], tmp_v[count+4]), 'first'])
                    val_dict[patterns[-1][0]].append(word)
                except Exception:
                    logging.exception('Exception in the discovery of patterns for {}'.format(word))
                    continue
    logging.info('We have discovered {} patterns'.format(len(patterns)))           
    return patterns, val_dict 

def extract_patterns(df, pattern_list, extract_context_words=False):
    if extract_context_words == False:
        extracted_qualities_dict = defaultdict(list)
        hierarch_relation_dict = defaultdict(list)
        multi_term_val_dict = defaultdict(list)
        multi_term_qualities_list = []
        context_pattern_list = []
        pattern_list.extend(PATTERN_LIST)
        pattern_list.extend(HEARST_PATTERN)
    else:
        context_word_dict = defaultdict(list)
        pattern_list = [list(e) for e in set([tuple(e) for e in pattern_list])]
    pattern_list = [[re.compile(p[0]), p[1]] for p in pattern_list]
    for k, v in tqdm(df.items()):
        logging.info("Currently processing {} PATTERNS".format(len(pattern_list)))
        for pattern in pattern_list:
            if extract_context_words == False:
                logging.debug("Currently extracting pattern: {}".format(pattern))
                extracted_qualities_list, hierarch_relation_list, context_list = extract_term(pattern[0], pattern[1], v)
                if context_list == []:
                    logging.debug('ADDING TERM {} EXTRACTED WITH PATTERN: {}'.format(extracted_qualities_list, pattern[0]))
                    if is_nested_list(hierarch_relation_list) == True:
                        hierarch_relation_dict[k].extend([*chain.from_iterable(hierarch_relation_list)])   
                    if is_nested_list(extracted_qualities_list) == True:
                        extracted_qualities_dict[k].extend([*chain.from_iterable(extracted_qualities_list)])
                    else:
                        extracted_qualities_dict[k].extend(extracted_qualities_list)
                        hierarch_relation_dict[k].extend(hierarch_relation_list)
                else:
                    context_pattern_list.extend(context_list)
                    multi_term_val_dict[k].extend(extracted_qualities_list)
                    multi_term_qualities_list.extend(extracted_qualities_list)
            else:
                 context_word_dict = merge_dict(context_word_dict, extract_context(pattern[0], pattern[1], v))
    if extract_context_words == False:
        #logging.debug("VALIDATE_MULTI_TERMS: {}".format(validate_multi_terms(multi_term_val_dict, {})))
        print("Validating Multi-terms...")
        c_value_dict = validate_multi_terms(multi_term_val_dict, {})
        #mean_c_value = np.mean([v for v in c_value_dict.values()])
        #logging.debug('MEAN_C_VALUE: {}'.format(mean_c_value))
        #logging.debug('C_VALUE_DICT.VALUES(): {}'.format(c_value_dict.values()))
        logging.debug("C_VALUE_DICT.KEYS(): {}".format(len(c_value_dict.keys())))
        top_n_multi_term_qualities_list = nlargest(600, c_value_dict, c_value_dict.get) #if v > mean_c_value]
        return extracted_qualities_dict, multi_term_val_dict, top_n_multi_term_qualities_list, hierarch_relation_dict, context_pattern_list
    else:
        return context_word_dict

def extract_term(compiled_pattern, term_loc, s):
    tmp_rslt = compiled_pattern.findall(s)
    if term_loc == 'first':
        return [i.split(' ')[0] for i in tmp_rslt], [], [] 
    if term_loc == 'last':
        return [i.split(' ')[:-1] for i in tmp_rslt], [], []
    if term_loc == 'hfirst':
        logging.debug('HFIRST: {}'.format(tmp_rslt))
        if tmp_rslt == []:
            return [], [], []
        else:
            tmp_rslt_list = [i for i in [*chain.from_iterable(tmp_rslt)][0].split(' ') if i.startswith('NOUN')]
            out = []
            for i in tmp_rslt_list[1:]:
                tmp_term = '{}~{}'.format(tmp_rslt_list[0], i)
                out.append(tmp_term)
                H_PATTERN_VAL[compiled_pattern].append(tmp_term)
            return tmp_rslt_list, out, []
    if term_loc == 'hlast':
        logging.debug('HLAST: {}'.format(tmp_rslt))
        if tmp_rslt == []:
            return [], [], []
        else:
            tmp_rslt_list = [i for i in [*chain.from_iterable(tmp_rslt)][0].split(' ') if i.startswith('NOUN')]
            out = []
            for i in tmp_rslt_list[0:-1]:
                tmp_term = '{}~{}'.format(tmp_rslt_list[:-1], i[0])
                out.append(tmp_term)
                H_PATTERN_VAL[compiled_pattern].append(tmp_term)
            return tmp_rslt_list, out, []
    if term_loc == 'rfull':
        print("TMP_RSLT: {}".format(tmp_rslt))
        logging.debug("RFULL: {}".format(tmp_rslt))
        if tmp_rslt == []:
            return [], [], []
        else:
            print("ELSE TMP_RSLT: {}".format(tmp_rslt))
            tmp_rslt_list = [e[0] for e in tmp_rslt if e[0] != ''] #[i for i in [*chain.from_iterable(tmp_rslt)][0].split(' ')]
            print("ELSE II TMP_RSLT_LIST: {}".format(tmp_rslt_list))
            return tmp_rslt_list, [], []
    else:
        out = []
        context_pattern_list = []
        for e in tmp_rslt:
            try:
                out.append(e[0])
                context_pattern_list.append([r'((NOUN\w+|ADJ\w+|VERB\w+)\s{})'.format(e[0]), 'last'])
                context_pattern_list.append([r'({}\s(NOUN\w+|ADJ\w+|VERB\w+))'.format(e[0]), 'first'])
            except:
                out.append(e)
                context_pattern_list.append([r'((NOUN\w+|ADJ\w+|VERB\w+)\s{})'.format(e), 'last'])
                context_pattern_list.append([r'({}\s(NOUN\w+|ADJ\w+|VERB\w+))'.format(e), 'first'])
        return out, [], context_pattern_list
    
def extract_context(compiled_pattern, term_loc, s):
    context_word_dict = defaultdict(list)
    tmp_rslt = [e[0] for e in compiled_pattern.findall(s)]
    if term_loc == 'first':
        for i in tmp_rslt:
            context_word_dict[' '.join(i.split(' ')[1:])].append(i.split(' ')[0] for i in tmp_rslt)
        return context_word_dict
    else:
        for i in tmp_rslt:
            context_word_dict[' '.join(i.split(' ')[0:-1])].append(i.split(' ')[:-1] for i in tmp_rslt)
        return context_word_dict

def validate_patterns(val_dict, pattern_list, seed_list):
    patterns = []
    for pattern in pattern_list:
        if pattern[1] in ['first', 'last']:
            if get_est_recall(val_dict, pattern[0], seed_list) > 1/len(seed_list): 
                if pattern not in patterns:
                    patterns.append(pattern)
                else:
                    continue
        else:
            patterns.append(pattern)
    return patterns 

def get_freq_dict(extracted_qualities_dict):
    return Counter([*chain.from_iterable([v for v in extracted_qualities_dict.values()])])

def get_substring_freq_dict(term, extracted_qualities_list, freq_dict):
    super_term_list = []
    for e in extracted_qualities_list:
        if freq_dict.get(e) == None:
            continue 
        else:
            if len(term) > len(e):
                break 
            else:
                if term in e:
                    super_term_list.append(e)
                else:
                    continue
    if len(super_term_list) > 0:
        return len(super_term_list), sum([freq_dict[e] for e in super_term_list]) 
    else:
        return 0, 0

def get_n_words(s):
    return len(s.split(' '))

def get_c_value(term, extracted_qualities_dict, freq_dict):
    extracted_qualities_list = [*np.unique([*chain.from_iterable([v for v in extracted_qualities_dict.values()])])]
    extracted_qualities_list.sort(reverse=True)
    tmp_n, tmp_f = get_substring_freq_dict(term, extracted_qualities_dict, freq_dict)
    logging.debug('SUBSTRING_FREQ_DICT for {}: {} and {}'.format(term, tmp_n, tmp_f))
    if tmp_n == 0:
        return math.log2(get_n_words(term)) * freq_dict[term]
    else:
        return math.log2(get_n_words(term)) * (freq_dict[term] - (1 / (tmp_n * tmp_f)))

def get_weight(total_n_terms_for_context_term, total_n_terms):
    return  total_n_terms_for_context_term / total_n_terms

def get_nc_value(candidate_term, context_word_dict, c_value_dict):
    context_weight_dict = defaultdict()
    word_context_dict = invert_dict(context_word_dict)
    [context_weight_dict.update({v:get_weight(len(set(word_context_dict[v])), len(context_word_dict.keys()))}) for v in context_word_dict[candidate_term]]
    freq_dict = Counter(context_word_dict[candidate_term])
    summed_weights = sum([freq_dict[v] * context_weight_dict[v] for v in [*set(context_word_dict[candidate_term])]])
    return (0.8 * c_value_dict[candidate_term] + 0.2 * summed_weights) 

def validate_multi_terms(extracted_qualities_dict, context_word_dict, c_value=True):
    #c_value_dict = defaultdict(float)
    #nc_value_dict = defaultdict(float)
    wq_dict = defaultdict(list)
    freq_dict = {k:v for k, v in get_freq_dict(extracted_qualities_dict).items() if v > 3}
    freq_term_list = list(freq_dict.keys())
    freq_term_list.sort()
    logging.debug('FREQ_DICT: {}'.format(get_freq_dict(extracted_qualities_dict)))
    c_value_dict = sort_dict_by_values({k:get_c_value(k, extracted_qualities_dict, freq_dict) for k in tqdm(freq_term_list)})
    logging.debug('C_VALUE_DICT: {}'.format(c_value_dict))
    #[c_value_dict.update({k:get_c_value(k, extracted_qualities, freq_dict)}) for k in freq_term_list]
    #c_value_dict = sort_dict_by_values(c_value_dict)
    if c_value ==True:
        return c_value_dict
    else:
        #[nc_value_dict.update({k:get_nc_value(k, context_word_dict, c_value_dict)}) for k in c_value_dict.keys()] 
        #nc_value_dict = {k:get_nc_value(k, context_word_dict, c_value_dict) for k in c_value_dict.keys()}
        min_c_value = [*c_value_dict.values()]
        min_c_value.sort(reverse=True)
        min_c_value = min(min_c_value[0:299])
        nc_value_dict = {k:get_nc_value(k, context_word_dict, c_value_dict) for k, v in c_value_dict.keys() if v > min_c_value}
        [[wq_dict[k].append(e) for e in v if nc_value_dict[e] > 0] for k, v in extracted_qualities_dict.items()]
        logging.debug("NC_VALUE_DICT: {}".format(nc_value_dict))
        return wq_dict

def get_est_precision(alpha=1):
    return NotImplementedError

def get_est_recall(val_dict, pattern, seed_list):
    return len(np.unique(val_dict[pattern])) / len(seed_list)

def construct_context_patterns(pattern_list):
    context_pattern_list = []
    for pattern in pattern_list:
        context_pattern_list.append()

def validate_h_patterns(h_rels_dict):
    """
    Method for the validation of the extracted hearst patterns. For robustness of results, we chose to extract
    candidate terms only when they were matched by at least two different patterns. Furthermore, the relations are
    checked such that if p(y, x) < p(x, y), then we exclude p(y, x).
    """
    out = []
    h_rels_ls = [*chain.from_iterable([v for v in h_rels_dict.values() if len(set(H_PATTERN_VAL)) > 1])]
    h_rels_prob_dict = get_p(h_rels_ls)
    for rel in h_rels_ls:
        tmp_ls = rel.split('~')
        tmp_inv_rel = '{}~{}'.format(tmp_ls[1], tmp_ls[0])
        if h_rels_prob_dict.get(tmp_inv_rel) == None:
            h_rels_prob_dict[tmp_inv_rel] = 0
        logging.debug("REL: {} and INV_REL: {}".format(h_rels_prob_dict[rel], h_rels_prob_dict[tmp_inv_rel]))
        if h_rels_prob_dict[rel] > h_rels_prob_dict[tmp_inv_rel]:
            out.append(rel)
        else:
            continue
    return out
