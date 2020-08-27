"""
An evolving tool for computing info from actual clusters
"""
import argparse  
import os
import pandas as pd
import pickle
import sys
import traceback

import pickle

from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from operator import itemgetter

# ---------------------------------------------
#        a bit of tools
# ---------------------------------------------

# just for custom printing (trace purpose)
class FeCounter(Counter):
    def __str__(self):
        return " ".join(f'{k}' for k, v in self.items() if v > 1)

# ---------------------------------------------
#        gestion ligne de commande
# ---------------------------------------------

def get_args():

    parser = argparse.ArgumentParser(description='analyse clusters')
    parser.add_argument("input_file", help="A pickle input file, e.g. aircan-data-split-clean.pkl.")

    parser.add_argument("-v", '--verbosity', type=int, help="increase output verbosity", default=0)
    parser.add_argument("-t", '--test', action='store_true', help="for dealing with test", default=False)
    parser.add_argument("-p", '--pickle', type=str, help="Pickle Bows", default=None)
    

    args = parser.parse_args()
    return args

# ---------------------------------------------
#        main
# ---------------------------------------------


def main():

    # parse args
    args = get_args()

    if not os.path.exists(args.input_file):
        print(f"Invalid input file: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # read data; this will load the data as 6 pandas DataFrames, which allow fast manipulations and (slower) iterations
    # more info on pandas here: https://pandas.pydata.org/
    try:
        with open(args.input_file, 'rb') as fin:
            [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)
            print(f"Read # samples: {len(defect_df_train)} train, {len(defect_df_test)} dev, {len(defect_df_test)} test.")
    except:
        print("Loading the pickle failed.", file=sys.stderr)

        if pd.__version__ != '1.1.0':
            print("""You can upgrade your version of pandas with the command 
                  'pip install 'pandas==1.1.0' --force-reinstall'.""", file=sys.stderr)

        print("""You can also recreate the pickle by following the instructions here: 
                 https://github.com/rali-udem/arpi_air_canada#data-preparation""", file=sys.stderr)
        print()
        traceback.print_exc()

    # basic stats on each cluster
    check_ref_clusters(defect_df_test if args.test is True else defect_df_train, args.pickle)



# ---------------------------------------------------------------
# felipe's function (to ramp up on pandas I never used seriously)
# incidentally producing views of clusters
# ---------------------------------------------------------------

def check_ref_clusters(defect, save):

    # ATA-signature -> bow (counter)
    bows = {}

    # note: ATA signatures might be null, which might generate some noise (even bugs)
    #       for now I leave it like this

    grouped_by_recurrent = defect.groupby('recurrent')
    for name, group in grouped_by_recurrent:

       l = len(group)
       if l == 1:
           #ignore clusters with only one member (it does happen !)
           print(f"#WARNING: recurrent defect {name} has only one member (skipped)")
       else:

           # the count the number of chapter-section signatures per cluster 
           # (mind you: some clusters have numerous signatures, which defeats my understanding of TRAX)
           grouped_by_ata = group.groupby(['chapter', 'section'])
           print(f"---\n#INFO: Recurrent defect {name}, with {len(group)} member(s), and  {len(grouped_by_ata)} ata-code(s)")
           if len(grouped_by_ata) > 1:
                # warn if more than one signature
                print(f"#WARNING: more than one chapter-section ({len(grouped_by_ata)})")

           # number of lines retained in the cluster 
           nb = 0

           # let's keep track of words in a given cluster
           c = FeCounter()

           # iterate over signatures in the cluster
           for sname,sgroup in grouped_by_ata:
                code = format(f"{sname[0]}-{sname[1]}")
                print(f"+ ata-code: {code}")
                if sname[1] != 0:
                    # and print concerned lines provided section is not 0 and the description is filled
                    for index,row in sgroup.iterrows():
                        desc = row['defect_description']
                        if pd.notnull(desc): 
                          c.update(word_tokenize(desc.lower()))
                          print(f"\t#line\t{index}\t{row['chapter']}-{row['section']}\t{row['ac']}\t{desc}")                          
                          nb += 1

                    if code not in bows: 
                      bows[code] = Counter(c)
                    else:
                      bows[code].update(c)

           # cluster-wise journalization              
           print(f"#trace: {nb} safe lines for defect {name}")
           print("#bow: ",c)

    # dataset-wise journalization
    print(f"#ata-signatures: {len(bows)}")
    for signature,bow in bows.items():
      b = bow.most_common(10)
      #b = dict(sorted(bow.items(), key=itemgetter(1), reverse=True))
      print(f"#bow({signature}) [{len(bow)}]: {b}")

    if not save is None:
      outfile = open(save,'wb')
      pickle.dump(bows,outfile)
      outfile.close()
      print(f"Generated pickle: {save}")


      
if __name__ == '__main__':
    main()
