"""Just quick relabeling stats when working with reliable dataset"""
import argparse
import pickle

import pandas as pd

import arpi_evaluator


def main():
    # parse args
    parser = argparse.ArgumentParser("A sample program to test text normalization.")
    parser.add_argument("input_file", help="A pickle input file, e.g. aircan-data-split-clean.pkl.")
    parser.add_argument("output_file", help="Output file, tsv format.")

    args = parser.parse_args()

    print("Loading...")
    with open(args.input_file, 'rb') as fin:
        [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)

    defect_df_full = pd.concat([defect_df_train, defect_df_test, defect_df_dev], sort=True)

    print("Relabeling...")
    relabeling_stats = arpi_evaluator.relabel_ata(defect_df_full)

    print("Writing stats in " + args.output_file + "...")
    with open(args.output_file, 'wt', encoding='utf-8') as fout:
        for src_ata, trg_ata_list in relabeling_stats.items():
            for trg_ata in trg_ata_list.keys():
                print('\t'.join([str(x) for x in [src_ata, trg_ata, trg_ata_list[trg_ata]]]), file=fout)


if __name__ == '__main__':
    main()
