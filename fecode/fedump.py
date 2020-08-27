"""Dumps a column of a database"""
import argparse
import re
import sys

import numpy as np
import os
import pickle
import pandas as pd

#from nltk.tokenize import sent_tokenize, word_tokenize
  
# ---------------------------------------------
#       globals
# ---------------------------------------------
 
BASES = {'train': 0, 'dev': 1, 'test': 2, 'ata': 3, 'mel': 4, 'trax': 5 }

# ---------------------------------------------
#       ligne de commande
# ---------------------------------------------
# 
def get_args(): 
    parser = argparse.ArgumentParser(description='search for (acro) in descriptions.')
    
    parser.add_argument("input_file", help="A pickle input file, e.g. aircan-data-split-clean.pkl.")
    parser.add_argument("-v", '--verbosity', type=int, help="increase output verbosity", default=0) 
    parser.add_argument("-b", '--base', type=str, choices=BASES.keys(), help="the name of the df", default='trax') 
    parser.add_argument("-c", '--column', type=str, help="the name of the df", default='defect_description') 
 
    args = parser.parse_args()
    return args

# ---------------------------------------------
#        main
# ---------------------------------------------
def main():
 
    global BASES

    args = get_args()
     
    # open dfs
    with open(args.input_file, 'rb') as fin:
        dfs = pickle.load(fin)
        [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = dfs

    # remove empty descriptions
    clean = dfs[BASES[args.base]].dropna(subset=[args.column])

    print(f"base: {args.base} column: {args.column} #{len(clean)}",file=sys.stderr)

    # iterate through the df (likely an easier way)
    for index, row in clean.iterrows():
        print(row[args.column])

# ---------------------------------------------
#        
# ---------------------------------------------
if __name__ == '__main__':
    main()
