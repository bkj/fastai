#!/usr/bin/env python

"""
    main.py
    
    Simplification of ULMFit
"""

from fastai.core import to_np, to_gpu, T, partition_by_cores, children
from fastai.dataloader import DataLoader
from fastai.dataset import ModelData

from fastai.lm_rnn import seq2seq_reg, get_language_model, get_rnn_classifer
from fastai.text import Tokenizer, TextDataset, SortSampler, SortishSampler

import re
import html
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F

# --

BOS  = 'xbos'  # beginning-of-sentence tag
FLD  = 'xfld'  # data field tag
PATH = Path('data/aclImdb/')

CLAS_PATH = Path('data/imdb_clas/')
CLAS_PATH.mkdir(exist_ok=True)

LM_PATH = Path('data/imdb_lm/')
LM_PATH.mkdir(exist_ok=True)

# --
# Format data

CLASSES = ['neg', 'pos', 'unsup']
COL_NAMES = ['labels', 'text']

def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r').read())
            labels.append(idx)
    return np.array(texts),np.array(labels)

trn_texts, trn_labels = get_texts(PATH/'train')
val_texts, val_labels = get_texts(PATH/'test')

# Classifier data
np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))

trn_texts  = trn_texts[trn_idx]
trn_labels = trn_labels[trn_idx]
df_trn     = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=COL_NAMES)
df_trn[df_trn['labels']!=2].to_csv(CLAS_PATH/'train.csv', header=False, index=False)

val_texts  = val_texts[val_idx]
val_labels = val_labels[val_idx]
df_val     = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=COL_NAMES)
df_val.to_csv(CLAS_PATH/'test.csv', header=False, index=False)

(CLAS_PATH/'classes.txt').open('w').writelines(f'{o}\n' for o in CLASSES)

# LM data
# !! Training the LM on train and test data?  Smells a little funny to me...
# !! Not a huge deal probably, 
`
trn_texts,val_texts = train_test_split(
    np.concatenate([trn_texts,val_texts]), test_size=0.1)

df_trn = pd.DataFrame({'text':trn_texts, 'labels':[0]*len(trn_texts)}, columns=COL_NAMES)
df_val = pd.DataFrame({'text':val_texts, 'labels':[0]*len(val_texts)}, columns=COL_NAMES)

df_trn.to_csv(LM_PATH/'train.csv', header=False, index=False)
df_val.to_csv(LM_PATH/'test.csv', header=False, index=False)

# --
# Language model tokens

chunksize = 24000
max_vocab = 60000
min_freq  = 2

re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)):
        texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    
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

df_trn = pd.read_csv(LM_PATH/'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(LM_PATH/'test.csv', header=None, chunksize=chunksize)

tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)

(LM_PATH/'tmp').mkdir(exist_ok=True)
np.save(LM_PATH/'tmp'/'tok_trn.npy', tok_trn)
np.save(LM_PATH/'tmp'/'tok_val.npy', tok_val)

tok_trn = np.load(LM_PATH/'tmp'/'tok_trn.npy')
tok_val = np.load(LM_PATH/'tmp'/'tok_val.npy')

freq = Counter(p for o in tok_trn for p in o)
freq.most_common(25)

itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')

stoi = defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)

trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi[o] for o in p] for p in tok_val])

np.save(LM_PATH/'tmp'/'trn_ids.npy', trn_lm)
np.save(LM_PATH/'tmp'/'val_ids.npy', val_lm)
pickle.dump(itos, open(LM_PATH/'tmp'/'itos.pkl', 'wb'))

trn_lm = np.load(LM_PATH/'tmp'/'trn_ids.npy')
val_lm = np.load(LM_PATH/'tmp'/'val_ids.npy')
itos = pickle.load(open(LM_PATH/'tmp'/'itos.pkl', 'rb'))

n_tok = len(itos)
n_tok, len(trn_lm)

# --
# Load wikitext103

emb_sz, nhid, nlayers = 400, 1150, 3
PRE_PATH = Path('data')/'models'/'wt103'
PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'

wgts     = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)
enc_wgts = to_np(wgts['0.encoder.weight'])
row_m    = enc_wgts.mean(0)

itos2 = pickle.load((PRE_PATH/'itos_wt103.pkl').open('rb'))
stoi2 = defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})

new_w = np.zeros((n_tok, emb_sz), dtype=np.float32)
for i,w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r >= 0 else row_m

wgts['0.encoder.weight']                    = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight']                    = T(np.copy(new_w))

# --
# Language model

# Dataloader
class LanguageModelLoader():
    def __init__(self, nums, bs, bptt, backwards=False):
        self.bs        = bs
        self.bptt      = bptt
        self.backwards = backwards
        self.data      = self.batchify(nums)
        self.i         = 0
        self.iter      = 0
        self.n         = len(self.data)
        
    def __iter__(self):
        self.i,self.iter = 0,0
        while self.i < self.n-1 and self.iter<len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res
            
    def __len__(self):
        return self.n // self.bptt - 1
        
    def batchify(self, data):
        nb = data.shape[0] // self.bs
        data = np.array(data[:nb*self.bs])
        data = data.reshape(self.bs, -1).T
        if self.backwards:
            data = data[::-1]
        return T(data)
        
    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)


class RNN_Encoder(nn.Module):
    
    """A custom RNN encoder network that uses
        - an embedding matrix to encode input,
        - a stack of LSTM layers to drive the network, and
        - variational dropouts in the embedding and LSTM layers
        
        The architecture for this network was inspired by the work done in
        "Regularizing and Optimizing LSTM Language Models".
        (https://arxiv.org/pdf/1708.02182.pdf)
    """
    
    def __init__(self, n_tok, emb_sz, nhid, nlayers, pad_token, bidir=False,
                 dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5, initrange=0.1):
        """ Default constructor for the RNN_Encoder class
            
            Args:
                bs (int): batch size of input data
                n_tok (int): number of vocabulary (or tokens) in the source dataset
                emb_sz (int): the embedding size to use to encode each token
                nhid (int): number of hidden activation per LSTM layer
                nlayers (int): number of LSTM layers to use in the architecture
                pad_token (int): the int value used for padding text.
                dropouth (float): dropout to apply to the activations going from one LSTM layer to another
                dropouti (float): dropout to apply to the input layer.
                dropoute (float): dropout to apply to the embedding layer.
                wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.
                
            Returns:
                None
          """
          
        super().__init__()
        
        self.ndir = 2 if bidir else 1
        self.bs = 1
        
        self.encoder = nn.Embedding(n_tok, emb_sz, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        self.rnns = [
            nn.LSTM(
                input_size=emb_sz if l == 0 else nhid, 
                hidden_size=(nhid if l != nlayers - 1 else emb_sz) // self.ndir,
                num_layers=1, 
                bidirectional=bidir, 
                dropout=dropouth
            ) for l in range(nlayers)]
        
        if wdrop:
            self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        
        self.emb_sz    = emb_sz
        self.nhid      = nhid
        self.nlayers   = nlayers
        self.dropoute  = dropoute
        self.dropouti  = LockedDropout(dropouti)
        self.dropouths = nn.ModuleList([LockedDropout(dropouth) for l in range(nlayers)])
        
    def forward(self, input):
        """ Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input (Tensor): input of shape (sentence length x batch_size)
    
        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)): list of tensors evaluated from each RNN layer without using
            dropouth, list of tensors evaluated from each RNN layer using dropouth,
        """
        
        sl, bs = input.size()
        if bs != self.bs:
            self.bs = bs
            self.reset()
        
        emb = self.encoder_with_dropout(input, dropout=self.dropoute if self.training else 0)
        emb = self.dropouti(emb)
        
        raw_output = emb
        new_hidden, raw_outputs, outputs = [], [], []
        for l, (rnn, drop) in enumerate(zip(self.rnns, self.dropouths)):
            current_input = raw_output
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_output, new_h = rnn(raw_output, self.hidden[l])
            
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            
            if l != self.nlayers - 1:
                raw_output = drop(raw_output)
            outputs.append(raw_output)
            
        self.hidden = repackage_var(new_hidden)
        return raw_outputs, outputs
        
    def one_hidden(self, l):
        nh = (self.nhid if l != self.nlayers - 1 else self.emb_sz)//self.ndir
        return Variable(self.weights.new(self.ndir, self.bs, nh).zero_(), volatile=not self.training)
        
    def reset(self):
        self.weights = next(self.parameters()).data
        self.hidden = [(self.one_hidden(l), self.one_hidden(l)) for l in range(self.nlayers)]


class LinearDecoder(nn.Module):
    def __init__(self, in_features, out_features, dropout, decoder_weights=None, initrange=0.1):
        super().__init__()
        
        self.decoder = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        if decoder_weights:
            self.decoder.weight = decoder_weights.weight
        
        self.dropout = LockedDropout(dropout)
    
    def forward(self, input):
        raw_outputs, outputs = input
        
        x = self.dropout(outputs[-1])
        x = x.view(x.size(0) * x.size(1), x.size(2))
        x = self.decoder(x)
        x = x.view(-1, x.size(1))
        
        return x, raw_outputs, outputs


class SequentialRNN(nn.Sequential):
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'):
                c.reset()


class LanguageModel():
    def __init__(self, n_tok, emb_sz, nhid, nlayers, pad_token,
                 dropout=0.4, dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, tie_weights=True):
        
        rnn_enc = RNN_Encoder(
            n_tok=n_tok,
            emb_sz=emb_sz,
            nhid=nhid,
            nlayers=nlayers,
            pad_token=pad_token,
            dropouth=dropouth,
            dropouti=dropouti,
            dropoute=dropoute,
            wdrop=wdrop,
        )
        
        self.model = SequentialRNN(*[
            rnn_enc,
            LinearDecoder(
                in_features=emb_sz,
                out_features=n_tok,
                dropout=dropout,
                decoder_weights=rnn_enc.encoder if tie_weights else None
            )
        ])
        self.model = to_gpu(self.model)
        
    def get_layer_groups(self):
        m = self.model[0]
        return [*zip(m.rnns, m.dropouths), (self.model[1], m.dropouti)]


class RNN_Learner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
    
    def _get_crit(self, data): 
        return F.cross_entropy
    
    def save_encoder(self, name):
        save_model(self.model[0], self.get_model_path(name))
    
    def load_encoder(self, name):
        load_model(self.model[0], self.get_model_path(name))


class LanguageModelData():
    def __init__(self, trn_dl, val_dl, test_dl=None):
        self.trn_dl  = trn_dl
        self.val_dl  = val_dl
        self.test_dl = test_dl


bptt  = 70
bs    = 52
drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7

learner = RNN_Learner(
    data=LanguageModelData(
        trn_dl=LanguageModelLoader(ds=np.concatenate(trn_lm), bs=bs, bptt=bptt),
        val_dl=LanguageModelLoader(ds=np.concatenate(val_lm), bs=bs, bptt=bptt),
    ), 
    models=LanguageModel(
        n_tok     = n_tok, 
        emb_sz    = emb_sz,
        nhid      = nhid, 
        nlayers   = nlayers, 
        pad_token = pad_token, 
        dropouti  = drops[0],
        dropout   = drops[1],
        wdrop     = drops[2],
        dropoute  = drops[3],
        dropouth  = drops[4],
    ), 
    opt_fn=partial(torch.optim.Adam, betas=(0.8, 0.99))
)

learner.metrics = [accuracy]
learner.freeze_to(-1)

learner.model.load_state_dict(wgts)

lrs = 1e-3
wd  = 1e-7
learner.fit(lrs=lrs/2, n_cycle=1, wds=wd, use_clr=(32,2), cycle_len=1)

learner.save('lm_last_ft')
learner.load('lm_last_ft')
learner.unfreeze()
learner.fit(lrs=lrs, n_cycle=1, wds=wd, use_clr=(20,10), cycle_len=15)

learner.save('lm1')
learner.save_encoder('lm1_enc')

# # --
# # Classifier

# class TextModel():
#     def __init__(self, model, name='unnamed'):
#         self.model = model
#         self.name  = name
    
#     def get_layer_groups(self):
#         m = self.model[0]
#         return [(m.encoder, m.dropouti), *zip(m.rnns, m.dropouths), (self.model[1])]


# trn_clas = np.load(CLAS_PATH/'tmp'/'trn_ids.npy')
# val_clas = np.load(CLAS_PATH/'tmp'/'val_ids.npy')

# trn_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'trn_labels.npy'))
# val_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'val_labels.npy'))

# bptt, emb_sz, nhid, nlayers, bs = 70, 400, 1150, 3, 48
# n_tok  = len(itos)
# opt_fn = partial(torch.optim.Adam, betas=(0.8, 0.99))

# min_lbl = trn_labels.min()
# trn_labels -= min_lbl
# val_labels -= min_lbl
# c = int(trn_labels.max()) + 1

# trn_ds   = TextDataset(trn_clas, trn_labels)
# val_ds   = TextDataset(val_clas, val_labels)
# trn_samp = SortishSampler(trn_clas, key=lambda x: len(trn_clas[x]), bs=bs//2)
# val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))
# trn_dl   = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
# val_dl   = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
# md       = ModelData(PATH, trn_dl, val_dl)

# # !!!!!!!!!!!
# dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])
# dps = np.array([0.4,0.5,0.05,0.3,0.4]) * 0.5

# m = get_rnn_classifer(
#     bptt      = bptt, 
#     max_seq   = 20*70, 
#     n_class   = c, 
#     n_tok     = n_tok, 
#     emb_sz    = emb_sz,
#     n_hid     = nhid,
#     n_layers  = nlayers,
#     pad_token = 1,
#     layers    = [emb_sz*3, 50, c],
#     drops     = [dps[4], 0.1],
#     dropouti  = dps[0],
#     wdrop     = dps[1],
#     dropoute  = dps[2],
#     dropouth  = dps[3]
# )

# opt_fn = partial(torch.optim.Adam, betas=(0.7, 0.99))

# learn         = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
# learn.reg_fn  = partial(seq2seq_reg, alpha=2, beta=1)
# learn.clip    = 25.
# learn.metrics = [accuracy]

# lr  = 3e-3
# lrm = 2.6
# # !!!!!!!!!!!
# lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])
# lrs = np.array([1e-4,1e-4,1e-4,1e-3,1e-2])

# # !!!!!!!!!!!
# wd = 1e-7
# wd = 0
# learn.load_encoder('lm2_enc')

# learn.freeze_to(-1)
# # learn.lr_find(lrs/1000)
# # learn.sched.plot()
# learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))

# learn.save('clas_0')
# learn.load('clas_0')
# learn.freeze_to(-2)
# learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))

# learn.save('clas_1')
# learn.load('clas_1')
# learn.unfreeze()
# learn.fit(lrs, 1, wds=wd, cycle_len=14, use_clr=(32,10))

# learn.sched.plot_loss()
# learn.save('clas_2')
