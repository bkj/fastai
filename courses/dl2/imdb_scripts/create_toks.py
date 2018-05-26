#!/usr/bin/env python

"""
    creat_toks.py
"""

import os
import argparse

from fastai.text import *
import html

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag
re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls):
    if len(df.columns) == 1:
        labels = []
        texts = f'\n{BOS} {FLD} 1 ' + df[0].astype(str)
        texts = texts.apply(fixup).values.astype(str)
    else:
        labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
        texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
        for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
        texts = texts.apply(fixup).values.astype(str)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    df_trn = pd.read_csv(os.path.join(args.indir, 'train.csv'), header=None, chunksize=24000)
    df_val = pd.read_csv(os.path.join(args.indir, 'test.csv'), header=None, chunksize=24000)
    
    tmpdir = os.path.join(args.indir, 'tmp')
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    
    tok_trn, trn_labels = get_all(df_trn, n_lbls=1)
    tok_val, val_labels = get_all(df_val, n_lbls=1)
    
    np.save(os.path.join(tmpdir, 'tok_trn.npy'), tok_trn)
    np.save(os.path.join(tmpdir, 'tok_val.npy'), tok_val)
    np.save(os.path.join(tmpdir, 'lbl_trn.npy'), trn_labels)
    np.save(os.path.join(tmpdir, 'lbl_val.npy'), val_labels)
    
    trn_joined = [' '.join(o) for o in tok_trn]
    mdl_fn = os.path.join(tmpdir, 'joined.txt')
    open(mdl_fn, 'w', encoding='utf-8').writelines(trn_joined)

