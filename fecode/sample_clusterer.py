"""
Sample program to show how to load the data and how to evaluate a (dummy) algorithm.
"""
import argparse
import arpi_evaluator
import numpy as np
import os
import pandas as pd
import pickle
import sys
import traceback
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances

from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

# ---------------------------------------------
#        from Fab
# ---------------------------------------------

__BEGINNING_OF_TIME = np.datetime64('1970-01-01T00:00:00')
__TIMEDELTA_HOUR = np.timedelta64(1, 'h')

# ---------------------------------------------
#        a bit of tools
# ---------------------------------------------

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
    check_ref_clusters(defect_df_test if args.test is True else defect_df_train)
    exit(0)

    # show how a dummy clusterer can be evaluated and further show how pandas can be used
    #print("\nPredicting recurrence clusters...\n")
    #test_predictions = find_recurrent_defects_naively(defect_df_test)

    #print("\nEvaluation\n")
    #eval_debug_info = arpi_evaluator.evaluate_recurrent_defects(defect_df_test, test_predictions)
    #print("System name\tHomog.\tCompl.\tV-meas.\tARI\tComment")
    #print("%s\t%.2f\t%.2f\t%.2f\t%.3f\t%s" % ('Dummy System', eval_debug_info['homogeneity'],
    #                                          eval_debug_info['completeness'], eval_debug_info['v_measure'],
    #                                          eval_debug_info['ari_score'], 'Sample system only.'))

    #print(f"\nDumping debug info in file {args.output_file}")
    #with open(args.output_file, 'wt', encoding='utf-8') as fout:
    #    arpi_evaluator.dump_debug_info(defect_df_test, eval_debug_info, fout)


# ---------------------------------------------------------------
# felipe's function (to ramp up on pandas I never used seriously)
# incidentally producing views of clusters
# ---------------------------------------------------------------

def check_ref_clusters(defect):

    # iterate over all the table
    #for index,row in defect_df_train.iterrows():
    #    print(index,row['ac'], row['recurrent'])

    grouped_by_recurrent = defect.groupby('recurrent')
    for name, group in grouped_by_recurrent:
       l = len(group)
       if l == 1:
           print(f"#WARNING: recurrent defect {name} has only one member (skipped)")
       else:
           grouped_by_ata = group.groupby(['chapter', 'section'])
           print(f"---\n#INFO: Recurrent defect {name}, with {len(group)} member(s), and  {len(grouped_by_ata)} ata-code(s)")
           if len(grouped_by_ata) > 1:
                print(f"#WARNING: more than one chapter-section ({len(grouped_by_ata)})")
           nb = 0
           c = FeCounter()
           for sname,sgroup in grouped_by_ata:
                code = format(f"{sname[0]}-{sname[1]}")
                print(f"+ ata-code: {code}")
                if sname[1] != 0:
                    for index,row in sgroup.iterrows():
                        nb += 1
                        c.update(word_tokenize(row['defect_description'].lower()))
                        print(f"\t#line\t{index}\t{row['chapter']}-{row['section']}\t{row['ac']}\t{row['defect_description']}")
           print(f"#trace: {nb} safe lines for defect {name}")
           print("#bow: ",c)

# ---------------------------------------------------------------
# Fabrizio's code
# ---------------------------------------------------------------

def little_demo(defect_df_dev, defect_df_test, defect_df_train, ata_df, mel_df, trax_df):

    # show a few things possible with the pandas data frames for those who are not familiar
    print(f"\nThere are {len(defect_df_train.columns)} columns in all 3 defect dataframes, whose names and types are:")
    print(defect_df_train.dtypes)
    print("\nThe first few rows for train are:")
    print(defect_df_train.head())
    print(f"\nThere are {len(defect_df_train.ac.unique())} unique aircrafts in train, "
          f"{len(defect_df_dev.ac.unique())} in dev and {len(defect_df_test.ac.unique())} in test.")

    # show how to find fields by integer indices or by id
    print(f"The 3rd defect for dev: {defect_df_dev.iloc[2]}")
    description_column_index = defect_df_dev.columns.get_loc('defect_description')
    print("The 3rd defect for dev (only text portion): "
          f"{defect_df_dev.iloc[2, description_column_index]}")
    print(f"\nLookup a defect by id (L-5747551-1), then field (ac): {defect_df_train.loc['L-5747551-1']['ac']}")
    print(f"Is the value L-5747551-1 present in train?: {'L-5747551-1' in defect_df_train.index}")
    print(f"Is the value L-5747551-1 present in dev?: {'L-5747551-1' in defect_df_test.index}")

    print(defect_df_train.info())  # also fun

    # you can also lookup various info in the ata, mel and trax dataframes
    sample_defect = defect_df_test.loc['L-5531638-1']
    ata_key = (sample_defect.chapter, sample_defect.section)
    ata_value = ata_df.loc[ata_key]
    print(f"Description of ATA chapter/section {ata_key} is '{ata_value.description}'")

    print(f"Is MEL number {sample_defect.mel_number} in mel table?: {sample_defect.mel_number in mel_df}")

# ---------------------------------------------------------------
# ---------------------------------------------------------------

def find_recurrent_defects_naively(defect_df):
    """
    Finds recurrent defects (naive algorithm to show how to proceed with the data).

    :param defect_df: The defect dataframe for which we try to find recurrent defects.
    :return: A result datastructure in the format expected for evaluation.
    """
    result = []

    # we regroup defects by aircraft ('ac') since a given defect cannot be recurrent across different aircraft
    grouped_by_ac = defect_df.groupby('ac')

    # we iterate over each aircraft group, with a tuple (name of aircraft, dataframe for all defects for this aircraft)
    for name, ac_group in grouped_by_ac:
        print(f"Working on aircraft {name}, with {len(ac_group)} defects reported.")

        labels = []  # we prepare the labels for each defect, in the order we encounter them
        feature_matrix = np.zeros((len(ac_group), 2))  # we have only 2 features: chapter and timestamp, ofc improveable

        # we can then iterate over all rows of the data and use the fields we want!
        row_number = 0
        for index, row in ac_group.iterrows():  # index is the row index and row is the data itself (a series)
            cur_type = row['defect_type']  # e.g. C, E or L
            cur_defect = row['defect']  # an int
            cur_item = row['defect_item']  # an int

            cur_id = f'{cur_type}-{str(cur_defect)}-{str(cur_item)}'  # this uniquely identifies a defect
            labels.append(cur_id)  # for our clustering algorithm (below)

            if pd.notnull(row['chapter']):  # some values, like the ATA chapter here can be null (i.e. not available)
                cur_chapter = row['chapter']
            else:
                cur_chapter = -1

            cur_reported_date = row['reported_datetime']

            # we convert the date to hours
            cur_reported_hours = (cur_reported_date - __BEGINNING_OF_TIME) // __TIMEDELTA_HOUR

            # we add the features to the feature matrix
            feature_matrix[row_number] = (cur_chapter, cur_reported_hours)
            row_number += 1

        # our simple algorithm performs agglomerative clustering - this is not important, it's just a demo
        clustering_model = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                                   distance_threshold=.1, linkage='average')
        dist_matrix = pairwise_distances(feature_matrix, None, metric=custom_distance_fun)
        clusters = clustering_model.fit_predict(dist_matrix)

        # we convert the clusters array([192,  192, 1193, ...,  247,  247,  357]) into a result data structure
        cluster_map = {}  # a mapping from cluster name to list of defects
        for i, cluster_label in enumerate(clusters):
            cluster_map[cluster_label] = cluster_map.get(cluster_label, set())
            cluster_map[cluster_label].add(labels[i])

        ac_result = list(cluster_map.values())
        result += ac_result

    return result


def custom_distance_fun(defect1, defect2):
    """This returns a simple distance function in interval [0,1] between 2 defects."""
    distance = 1

    # first check if they are within 30 days of each other
    delta_days = abs(defect2[0] - defect1[0]) / 24
    if delta_days <= 30:  # 30 days for our example
        delta_chapter = 0 if defect1[1] == defect2[1] else 1
        distance = 0.3 * (delta_days / 30.0) + 0.7 * delta_chapter

    return distance


if __name__ == '__main__':
    main()
