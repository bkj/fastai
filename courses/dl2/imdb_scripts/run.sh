#!/bin/bash

# run.sh

# Fetch data
mkdir -p data
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzvf aclImdb_v1.tar.gz && mv aclImdb data/aclImdb/

# Make sure you have spacy english model downloaded
python -m spacy download en

# Create `train.csv` and `test.csv`
python prep.py --indir data/aclImdb --outdir tmp

python create_toks.py imdb

python tok2id.py

# python train_tri_wt.py
mkdir -p data/imdb_clas/wt103/models
wget http://files.fast.ai/models/wt103/bwd_wt103.h5
wget http://files.fast.ai/models/wt103/bwd_wt103_enc.h5
wget http://files.fast.ai/models/wt103/fwd_wt103.h5
wget http://files.fast.ai/models/wt103/fwd_wt103_enc.h5
wget http://files.fast.ai/models/wt103/itos_wt103.pkl
mv fwd* bwd* itos* data/imdb_clas/wt103/models

python train_tri_lm.py --cuda_id=0 --cl=5