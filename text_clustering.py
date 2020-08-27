from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

import pickle
import pandas
import arpi_evaluator
import preprocessing.spell_check as spell
from tqdm import tqdm


def map_clusters(model_labels: list, dataset: dict, train_split: tuple, ata_names: list):
    cluster2ata = dict()
    for i in range(train_split[0], train_split[1]):
        label = model_labels[i]
        target = ata_names[dataset['target'][i]]
        if target is not None:
            d = cluster2ata.get(label, dict())
            cluster2ata[label] = d
            n = d.get(target, 0)
            d[target] = n + 1

    for cluster, mapping in cluster2ata.items():
        print(cluster, len(mapping), mapping)
    return cluster2ata


def produce_recurrent_clustering(labels, dataset, split):
    recurrent_clusters = dict() # by aircraft

    last_timestamp = None
    last_ac = None
    for i in range(split[0], split[1]):
        defect_type, defect, defect_item, chapter, section, timestamp, ac, recurrent = dataset['features'][i]
        identifier = f'{defect_type}-{defect}-{defect_item}'
        label = labels[i]

        ac_clusters = recurrent_clusters.get(ac, dict())
        recurrent_clusters[ac] = ac_clusters

        candidate_cluster = ac_clusters.get(label, list())
        ac_clusters[label] = candidate_cluster

        candidate_cluster.append({'id': identifier, 'time': timestamp, 'r':recurrent})

    final_clusters = list()
    for ac, ac_clusters in recurrent_clusters.items():
        for label, candidate in ac_clusters.items():
            candidate = sorted(candidate, key=lambda e: e['time'])

            current = candidate.pop(0)
            while len(candidate) > 0:
                if len(candidate) >= 4 and candidate[3]['time'] - current['time'] <= pandas.Timedelta('5d'):
                     cluster = set([e['id'] for e in candidate[:4]])
                     cluster.add(current['id'])
                     final_clusters.append(cluster)
                     del candidate[:4]
                elif len(candidate) >= 3 and candidate[2]['time'] - current['time'] <= pandas.Timedelta('4d'):
                     cluster = set([e['id'] for e in candidate[:3]])
                     cluster.add(current['id'])
                     final_clusters.append(cluster)
                     del candidate[:3]
                elif len(candidate) >= 2 and candidate[1]['time'] - current['time'] <= pandas.Timedelta('3d'):
                     cluster = set([e['id'] for e in candidate[:2]])
                     cluster.add(current['id'])
                     final_clusters.append(cluster)
                     del candidate[:2]

                if len(candidate) > 0:
                    current = candidate.pop(0)

    return final_clusters


def process_row(col_name: str, row: dict, spelling: dict):
    txt = spell.process_txt(row[col_name])
    token_list = list()
    for token in txt.split():
        if spell.token_is_word_like(token):
            if token in spelling:
                token_list.append(spelling[token][0])
            else:
                token_list.append(token)
    row[col_name + '_tokens'] = token_list

    return token_list


def load_dataset(filename: str, spelling_filename: str = None):
    with open(filename, 'rb') as fin:
        [defect_df_train, defect_df_dev, defect_df_test, ata_df, mel_df, trax_df] = pickle.load(fin)

    defect_df_train.dropna(subset=['defect_description'], inplace=True)
    defect_df_train.dropna(subset=['resolution_description'], inplace=True)
    defect_df_dev.dropna(subset=['defect_description'], inplace=True)
    defect_df_dev.dropna(subset=['resolution_description'], inplace=True)
    defect_df_test.dropna(subset=['defect_description'], inplace=True)
    defect_df_test.dropna(subset=['resolution_description'], inplace=True)

    split = dict()
    split['train'] = (0, len(defect_df_train))
    split['dev'] = (split['train'][1], split['train'][1] + len(defect_df_dev))
    split['test'] = (split['dev'][1], split['dev'][1] + len(defect_df_test))

    defect_df = pandas.concat([defect_df_train, defect_df_dev, defect_df_test])
    assert(split['test'][1] == len(defect_df))
    arpi_evaluator.relabel_ata(defect_df)

    #arpi_evaluator.relabel_ata(defect_df_train)
    #recurrent_defects = dict()
    #for k, e in defect_df_train[['defect_type', 'defect', 'defect_item', 'ac', 'reliable_chapter', 'reliable_section', 'recurrent']].iterrows():
    #    if type(e['recurrent']) is int:
    #        d = recurrent_defects.get(e['recurrent'], dict())
    #        recurrent_defects[e['recurrent']] = d
    #        index = (e['reliable_chapter'], e['reliable_section'])
    #        if type(index[0]) is int and type(index[1]) is int:
    #            n = d.get(index, 0)
    #            d[index] = n + 1

    #for k, v in recurrent_defects.items():
    #    if len(v) > 1:
    #        print(k, v)
    #exit()

    spelling = dict()
    if spelling_filename is not None:
        spelling = spell.load_spell_dict(spelling_filename)

    data = defect_df.to_dict(orient='records')

    target = list()
    target_names = [None]
    features = list()
    index = dict()
    dataset_concat = { 'data': list(), 'target': target, 'target_names': target_names, 'features': features, 'index': index }
    dataset_defect = { 'data': list(), 'target': target, 'target_names': target_names, 'features': features, 'index': index }
    dataset_resolution = { 'data': list(), 'target': target, 'target_names': target_names, 'features': features, 'index': index }
    
    for row in tqdm(data):
        defect = " ".join(process_row('defect_description', row, spelling))
        resolution = " ".join(process_row('resolution_description', row, spelling))
        dataset_defect['data'].append(defect)
        dataset_resolution['data'].append(resolution)
        dataset_concat['data'].append(defect + ' ' + resolution)
        ident = f"{row['defect_type']}-{row['defect']}-{row['defect_item']}"
        assert(ident not in index)
        index[ident] = len(features)
        features.append((row['defect_type'], row['defect'], row['defect_item'], row['chapter'], row['section'], row['reported_datetime'], row['ac'], row['recurrent']))

        if type(row['reliable_chapter']) is int and type(row['reliable_section']) is int:
            ata = f"{row['reliable_chapter']} {row['reliable_section']}"  # also allow chapter only clustering?
            if ata in target_names:
                category = target_names.index(ata)
            else:
                category = len(target_names)
                target_names.append(ata)
        else:
            category = 0
        target.append(category)

    return dataset_defect, dataset_resolution, dataset_concat, split, (defect_df_train, defect_df_dev, defect_df_test)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")
op.add_option("--output", help="Ouptut log-like.")
op.add_option("--clustering-output", help="Where to output the best system's predicted clusters.")

#print(__doc__)
#op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


# #############################################################################
# Load some categories from the training set
#categories = [
#    'alt.atheism',
#    'talk.religion.misc',
#    'comp.graphics',
#    'sci.space',
#]
# Uncomment the following to do the analysis on all the categories
# categories = None

#print("Loading 20 newsgroups dataset for categories:")
#print(categories)

dataset_defect, dataset_resolution, dataset_concat, split, corpus = load_dataset("aircan-data-split-clean.pkl", "small_resources/spelling_full.txt")

def clustering(n_clusters, dataset, minibatch, ngrams):
    print("%d documents" % len(dataset['data']))
    print("%d categories" % len(dataset['target_names']))
    print()
    
    labels = dataset['target']
    true_k = np.unique(labels).shape[0]
    true_k = n_clusters
    
    print("Extracting features from the training dataset "
          "using a sparse vectorizer")
    t0 = time()
    if opts.use_hashing:
        if opts.use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english', alternate_sign=False,
                                       norm=None)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features,
                                           stop_words='english',
                                           alternate_sign=False, norm='l2')
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                     min_df=2, stop_words='english',
                                     use_idf=opts.use_idf, ngram_range=ngrams)
    X = vectorizer.fit_transform(dataset['data'])
    
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()
    
    if opts.n_components:
        print("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(opts.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
    
        X = lsa.fit_transform(X)
    
        print("done in %fs" % (time() - t0))
    
        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))
    
        print()
    
    
    # #############################################################################
    # Do the actual clustering
    
    if minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=opts.verbose)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1,
                    verbose=opts.verbose)
    
    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    #print("done in %0.3fs" % (time() - t0))
    #print()
    #
    #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    #print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    #print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    #print("Adjusted Rand-Index: %.3f"
    #      % metrics.adjusted_rand_score(labels, km.labels_))
    #print("Silhouette Coefficient: %0.3f"
    #      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
    #
    #print()
    
    
    if not opts.use_hashing:
        print("Top terms per cluster:")
    
        if opts.n_components:
            original_space_centroids = svd.inverse_transform(km.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    
        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()
    
    # cluster2ata = map_clusters(km.labels_, dataset, split['train'], dataset['target_names'])
    clusters = produce_recurrent_clustering(km.labels_, dataset, split['train'])
    
    return arpi_evaluator.evaluate_recurrent_defects(corpus[0], clusters), clusters


def export_clustering(filename: str, dataset: dict, clusters: list):
    mapping = ["<NA>"] * len(dataset['features'])
    print("Saving results...", end="")
    for i, c in enumerate(clusters):
        for ident in c:
            mapping[dataset['index'][ident]] = i

    with open(filename, 'w') as fout:
        for i in range(len(dataset['features'])):
            print_list = list(dataset['features'][i])
            print_list.append(mapping[i])
            print_list.append(dataset['data'][i])

            print("\t".join([str(e) for e in print_list]), file=fout)
    print("  done.")


out = sys.stdout
if opts.output is not None:
    out = open(opts.output, 'w')

datasets = [dataset_defect, dataset_resolution, dataset_concat]
results = list()
for minibatch in [False, True]:
    for ngrams in [(1,1), (1,2), (1,3), (2,2)]:
        for dataset_index in range(len(datasets)):
            for n_clusters in range(120, 480, 20):
                 full_eval, clusters = clustering(n_clusters, datasets[dataset_index], minibatch, ngrams)
                 params = {'minibatch': minibatch, 'ngrams': ngrams, 'dataset': dataset_index, 'n_clusters': n_clusters}
                 res = {'full_eval': full_eval, 'params': params}
                 results.append(res)
                 results = sorted(results, key=lambda e:e['full_eval']['completeness'], reverse=True)
                 best = dict()
                 best['ari_score'] = results[0]['full_eval']['ari_score']
                 best['homogeneity'] = results[0]['full_eval']['homogeneity']
                 best['completeness'] = results[0]['full_eval']['completeness']
                 best['v_measure'] = results[0]['full_eval']['v_measure']
                 print(f"Just ran {params}")
                 print(f"CURRENT BEST: {best} {results[0]['params']}", file=out)
                 if opts.clustering_output is not None and results[0] == res:
                     export_clustering(opts.clustering_output, datasets[dataset_index], clusters)
                 out.flush()

#print(eval_)
#print(eval_.keys())
#print(f"homogeneity {eval_['homogeneity']}")
#print(f"completeness {eval_['completeness']}")
#print(f"v_measure {eval_['v_measure']}")
#print(f"ari_score {eval_['ari_score']}")
