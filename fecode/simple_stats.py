"""Provide a few simple stats for the dataset"""

import argparse
import numpy as np
import pandas as pd
import pickle
import re
import sys

# parse args
parser = argparse.ArgumentParser("A sample program.")
parser.add_argument("input_file", help="A pickle input file, e.g. aircan-data-split-clean.pkl.")

args = parser.parse_args()

with open(args.input_file, 'rb') as fin:
    [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)
    print(f"Read # samples: {len(defect_df_train)} train, {len(defect_df_test)} dev, {len(defect_df_test)} test.")
    print(defect_df_test)

exit(0)
defect_df_full = pd.concat([defect_df_train, defect_df_test, defect_df_dev], sort=False)

print(f"Nb  of defects: {len(defect_df_full)}")
text_len = defect_df_full.defect_description.apply(lambda s: len(s) if not pd.isnull(s) else np.nan)
nb_toks = defect_df_full.defect_description.apply(lambda s: len(re.split(r'[\s,\.:;]', s)) if not pd.isnull(s) else np.nan)

print(f"Avg text len in chars: {text_len.mean():.1f}")
print(f"Avg text len in tokens: {nb_toks.mean():.1f} +- {nb_toks.std():.1f}")
