#!/usr/bin/env python

"""
    prep.py
"""

import os
import sys
import argparse
import numpy as np
from glob import glob
from fastai.text import *

# --
# Helpers

def get_texts(path):
    texts,labels = [],[]
    for idx, classname in enumerate(['neg', 'pos', 'unsup']):
        for fname in glob(os.path.join(path, classname, '*.*')):
            texts.append(open(fname, 'r').read())
            labels.append(idx)
    
    return np.array(texts), np.array(labels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='./data/aclImdb')
    parser.add_argument('--outdir', type=str, default='./data/imdb_clas')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

# --
# Run

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
    
    # --
    # Process train data
    
    print('loading %s/train' % args.indir, file=sys.stderr)
    trn_texts, trn_labels = get_texts(os.path.join(args.indir, 'train'))
    trn_idx    = np.random.permutation(len(trn_texts))
    trn_texts  = trn_texts[trn_idx]
    trn_labels = trn_labels[trn_idx]
    df_trn     = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=['labels','text'])
    df_trn     = df_trn[df_trn['labels']!=2]
    print('saving %s/train.csv' % args.outpath, file=sys.stderr)
    df_trn.to_csv(os.path.join(args.outpath, 'train.csv'), header=False, index=False)
    
    # --
    # Process test data
    
    print('loading %s/test' % args.indir, file=sys.stderr)
    val_texts, val_labels = get_texts(os.path.join(args.indir, 'test'))
    val_idx    = np.random.permutation(len(val_texts))
    val_texts  = val_texts[val_idx]
    val_labels = val_labels[val_idx]
    df_val     = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=['labels','text'])
    print('saving %s/test.csv' % args.outpath, file=sys.stderr)
    df_val.to_csv(os.path.join(args.outpath, 'test.csv'), header=False, index=False)




