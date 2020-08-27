"""Simple Levenshtein based spell checker"""

import argparse
import numpy as np
import pandas as pd
import pickle
import sys
import re
import math
from tqdm import tqdm

import multiprocessing
import queue
import time

MAX_EDIT_RATIO = 1/5
def capped_levenshtein(token: str, dictionary: set):
    tok_len = len(token)
    if tok_len * MAX_EDIT_RATIO < 1 or token in dictionary:
        return None

    candidates = list()
    for word in dictionary:
        cap = tok_len * MAX_EDIT_RATIO
        word_len = len(word)
        if abs(word_len - tok_len) > cap:
            continue

        tab = list()
        for i in range(tok_len+1):
            tab.append([i])
        for j in range(1, word_len+1):
            tab[0].append(j)

        for i in range(1, tok_len+1):
            local_minimum = max(tok_len, word_len)
            for j in range(1, word_len+1):
                edit = 1
                if token[i-1] == word[j-1]:
                    edit = 0
                current = min(tab[i-1][j]+1, tab[i][j-1]+1, tab[i-1][j-1]+edit)

                swap = 2
                if i > 1 and j > 1 and token[i-1] == word[j-2] and token[i-2] == word[j-1]:
                    swap = 1
                    current = min(current, tab[i-2][j-2]+swap)
                
                tab[i].append(current)
                local_minimum = min(local_minimum, tab[i][j])
            if local_minimum > cap:
                break

        if len(tab[tok_len]) == word_len+1 and tab[tok_len][word_len] <= cap:
            candidates.append((word, tab[tok_len][word_len]))

    candidates = sorted(candidates, key=lambda c:c[1])

    if len(candidates) == 0:
        return None
    elif len(candidates) == 1:
        return candidates[0][0]
    else:
        if candidates[0][1] == candidates[1][1]:  # ambiguity
            return None
        else:
            return candidates[0][0]


min_letters_pattern = re.compile(".*[a-zA-Z]{3}.*")
digit_pattern = re.compile(".*[0-9].*")
def token_is_word_like(token):
    if not min_letters_pattern.match(token):
        return False
    if digit_pattern.match(token):
        return False
    return True


def process_txt(txt):
    result = re.sub(r'[^a-zA-Z0-9]', ' ', txt)
    result = result.lower()
    return result


def spell_check(series, mpq, domain_dict: set, en_dict: set, acronyms: set):
    rows_empty = 0

    for txt in series:
        if type(txt) != str:
            rows_empty += 1
            continue
        txt = process_txt(txt)
        for token in txt.split():
            token = token.lower()

            if token.isalpha() and token not in en_dict and token not in domain_dict and token not in acronyms:
                result = capped_levenshtein(token, acronyms)
                if result is not None:
                    s = 'acronym'
                    if result in domain_dict:
                        s += '+data'
                    if result in en_dict:
                        s += '+dict'
                    mpq.put((token, result, s))
                else:
                    result = capped_levenshtein(token, domain_dict)
                    if result is not None:
                        if result in en_dict:
                            mpq.put((token, result, 'data+dict'))
                        else:
                            mpq.put((token, result, 'data'))
                    else:
                        result = capped_levenshtein(token, en_dict)
                        mpq.put((token, result, 'dict'))


"""
Use this function to load a precomputed output of this module for any use.
"""
def load_spell_dict(filename: str):
    spell_dict = dict()
    with open(filename) as fin:
        for line in fin.read().split('\n'):
            if line == '': # last line
                break
            [token, correction, confidence] = line.split('\t')
            assert(token not in spell_dict)
            spell_dict[token] = (correction, confidence)
    return spell_dict


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_input_file", help="A pickle input file, e.g. aircan-data-split-clean.pkl.")
    
    parser.add_argument("spell_check_output_file", help="Output file for the spell checking.")
    parser.add_argument("--en_dictionary", default="small_resources/en_dict.txt", help="Specify a different dictionary to check words against.")
    parser.add_argument("--acronyms", default="fecode/small_ressources/acronyms_full", help="An acronym list to exclude from corrections.")
    args = parser.parse_args()
    
    with open(args.corpus_input_file, 'rb') as fin:
        [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)
    
    defect_df_full = pd.concat([defect_df_train, defect_df_test, defect_df_dev], sort=False)

    en_dict = set()
    with open(args.en_dictionary) as fin:
        for word in fin.read().split():
            if len(word) * MAX_EDIT_RATIO >= 1:
                en_dict.add(word)

    acronyms = set()
    if args.acronyms is not None:
        with open(args.acronyms) as fin:
            for line in fin.read().split('\n'):
                if line != '':
                    acro = line.split()[1].lower()
                    if len(acro) * MAX_EDIT_RATIO >= 1:
                        acronyms.add(acro)

    domain_words = dict()
    for txt in defect_df_full.defect_description:
        if type(txt) != str:
            continue
        for word in process_txt(txt).split():
            word = word.lower()
            n = domain_words.get(word, 0)
            domain_words[word] = n + 1

    for txt in defect_df_full.resolution_description:
        if type(txt) != str:
            continue
        for word in process_txt(txt).split():
            word = word.lower()
            n = domain_words.get(word, 0)
            domain_words[word] = n + 1

    domain_dict = set()
    for word, n in domain_words.items():
        if word in en_dict and len(word) > 3:
            domain_dict.add(word)  # prioritizing dictionary words that appear in the data
        elif n > 100 and word.isalpha() and len(word) > 3:
            domain_dict.add(word)  # also prioritizing some frequent unknown tokens from the data

    mpq = multiprocessing.Queue()
    chunk_size = 100

    def writer(filename, mpq):
        with open(filename, 'w') as fout:
            results = dict()
            while True:
                try:
                    token, corrected, conf = mpq.get_nowait()
                    if token is None:
                        break
                    if token in results:
                        assert(results[token] == corrected)
                    elif corrected is not None:
                        print(f"{token}\t{corrected}\t{conf}", file=fout)
                        fout.flush()
                    results[token] = corrected
                except queue.Empty:
                    time.sleep(.5)
        
    max_jobs = max(multiprocessing.cpu_count() -2, 2)

    multiprocessing.Process(target=writer, args=(args.spell_check_output_file, mpq)).start()

    txt_series = defect_df_full.defect_description
    print("Searching corrections for defect descriptions...")
    for i in tqdm(range(0, len(txt_series), chunk_size)):
        while len(multiprocessing.active_children()) >= max_jobs:
            time.sleep(.5)
        multiprocessing.Process(target=spell_check, args=(txt_series[i:i+chunk_size], mpq, domain_dict, en_dict, acronyms)).start()

    txt_series = defect_df_full.resolution_description
    print("Searching corrections for resolution descriptions...")
    for i in tqdm(range(0, len(txt_series), chunk_size)):
        while len(multiprocessing.active_children()) >= max_jobs:
            time.sleep(.5)
        multiprocessing.Process(target=spell_check, args=(txt_series[i:i+chunk_size], mpq, domain_dict, en_dict, acronyms)).start()
    
    while len(multiprocessing.active_children()) > 2:
        time.sleep(.5)
    mpq.put((None, None, None))
