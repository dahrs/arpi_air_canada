"""
playing wiht wordclouds (course is approaching)
"""
import argparse  
import os
import pandas as pd
import pickle
import sys 
 
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
 
 
 
# ---------------------------------------------
#        gestion ligne de commande
# ---------------------------------------------

def get_args():

    parser = argparse.ArgumentParser(description='generate word clouds')
    parser.add_argument("input_file", help="A pickle input file, e.g. toto.pkl")

    #parser.add_argument("-v", '--verbosity', type=int, help="increase output verbosity", default=0)
    parser.add_argument("-k", '--key', type=str, help="chapter-section key", default="25-60")
    parser.add_argument("-m", '--min', type=float, help="min score to get in", default=None)
    parser.add_argument("-s", '--stop', action='store_true', help="wanna use a stop list?", default=False)
    parser.add_argument("-c", '--char', action='store_true', help="wanna filter single chars?", default=False)
    parser.add_argument("-d", '--dirty', action='store_true', help="wanna dirty normalization?", default=False)

    args = parser.parse_args()
    return args

# ---------------------------------------------
#        main
# ---------------------------------------------

def main():
    # parse args
    args = get_args()

    if args.stop:
        stopwords = set(STOPWORDS)
    else:
        stopwords = set()


    # read the pickle file
    infile = open(args.input_file,'rb')
    bows = pickle.load(infile)
    infile.close()

    # read wordcloud
    d = bows[args.key]

    # filter it (pipeline)
    if args.stop:
        # remove stopwords
        for word in stopwords:
            if word in d:
                del d[word]

    if not args.min is None:
        # remove score lower
        d = Counter({k: v for k, v in d.items() if v >= args.min})

    if args.char:
        # remove single chars
        d = Counter({k: v for k, v in d.items() if len(k) > 1})

    if args.dirty:
        # aggregate plural in s with the word without s if it exists
        u = {}
        for k,v in list(d.items()):
            if k[-1:] == 's' and k[:-1] in d:
                del d[k]
                u[k[:-1]] = v
        print(f"update: {u}")
        d.update(u)

    # generate
    print(f"generating wc with {len(d)} words")
    print(d)
    
    wc = WordCloud(background_color="white", max_font_size=100, relative_scaling=0.2).generate_from_frequencies(d)
    #wordcloud = WordCloud().generate("put your text here. Yes here !")


    # display
    plt.imshow(wc, interpolation='bilinear') 
    plt.axis("off")
    plt.title(f"ata-code: {args.key}")
   
    wc.to_file(f"img/{args.key}.png")
    plt.show()

 
# ---------------------------------------------
#         
# ---------------------------------------------
if __name__ == '__main__':
    main()

 