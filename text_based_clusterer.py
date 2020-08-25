import argparse
import pickle
import pandas
from tqdm import tqdm

import preprocessing.spell_check as spell
import preprocessing.ngrams
import arpi_evaluator


def extract_ngrams(n: int, tokens: list):
    if type(n) != int or n < 1:
        raise Exception("n must be a strictly positive integer")
    if len(tokens) < n:
        return list()

    ngrams = list()
    for i in range(len(tokens) +1 - n):
        ngrams.append(tuple(tokens[i:i+n]))

    return ngrams


def process_row(col_name: str, row: dict, document_counts: dict):
    txt = spell.process_txt(row[col_name])
    token_list = list()
    for token in txt.split():
        if spell.token_is_word_like(token):
            token_list.append(token)
    row[col_name + '_tokens'] = token_list

    for n, table in document_counts.items():
        for ngram in set(extract_ngrams(n, token_list)):
            count = table.get(ngram, 0)
            table[ngram] = count + 1
    return token_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="aircan-data-split-clean.pkl", help="A pickle input file.")
    parser.add_argument("--spelling", default="small_resources/spelling_full.txt", help="Precomputed spell check file.")
    
    args = parser.parse_args()
    ngram_choices = [1]  # token n-grams to consider

    with open(args.input_file, "rb") as fin:
        [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)

    spelling = dict()
    if args.spelling is not None:
        spelling = spell.load_spell_dict(args.spelling)
    
    arpi_evaluator.relabel_ata(defect_df_train)

    print(f"{len(defect_df_train)} rows in train")
    defect_df_train_notnull = defect_df_train[defect_df_train.defect_description.notnull()]
    print(f"{len(defect_df_train_notnull)} with a defect_description")
    defect_df_train_notnull = defect_df_train_notnull[defect_df_train_notnull.resolution_description.notnull()]
    print(f"{len(defect_df_train_notnull)} with a resolution_description")
    defect_df_train_with_mel = defect_df_train_notnull[defect_df_train_notnull.mel_number.notnull()]
    print(f"{len(defect_df_train_with_mel)} with a mel number")
#    defect_df_train_with_mel = defect_df_train[['defect_type', 'defect', 'defect_item', 'defect_description', 'resolution_description', 'mel', 'mel_number']]
#    defect_df_train_with_mel = defect_df_train_with_mel.join(mel_df, on='mel_number', rsuffix='_mel_df')

##    full_df = pandas.concat([defect_df_train, defect_df_dev, defect_df_test])
##    full_df = full_df[full_df.mel_number.notnull()]
##    data = full_df.to_dict(orient='index')
##    mel = mel_df.to_dict(orient='index')
##
##    unknown = dict()
##    for k, v in data.items():
##        if v['mel_number'] not in mel:
##            nb = unknown.get(v['mel_number'], 0)
##            unknown[v['mel_number']] = nb + 1
##
##    uk = list()
##    for k, v in unknown.items():
##        uk.append((k,v))
##    
##    with open("/tmp/unknown_mel_numbers.tsv", "w") as f:
##        for k, v in sorted(uk, key=lambda e:e[1]):
##            print(f"{k}\t{v}", file=f)
##    print(unknown)
##    exit()

    data = defect_df_train_notnull.to_dict(orient='index')
    mel = mel_df.to_dict(orient='index')

    data_with_mel = dict()
    ngram_defect_document_count = dict()
    ngram_resolution_document_count = dict()

    for table in [ngram_defect_document_count, ngram_resolution_document_count]:
        for n in ngram_choices:
            table[n] = dict()

    unknown_mel_numbers = set()
    
    for key, row in tqdm(data.items()):
        token_list = process_row('defect_description', row, ngram_defect_document_count)
#        token_list = process_row('resolution_description', row, ngram_resolution_document_count)

        row['mel_number'] = str(row['mel_number'])
        if row['mel_number'] in mel:
            data_with_mel[key] = row
        else:
            unknown_mel_numbers.add(row['mel_number'])

    for number in unknown_mel_numbers:
        print(f"{number} found in main dataframe but not in mel list")

    print(ngram_defect_document_count)

    # spell check texts

    # extract relevant n-grams

    # identify features
