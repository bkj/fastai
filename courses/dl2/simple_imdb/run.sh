#!/bin/bash


# Feth data
mkdir -p data
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzvf aclImdb_v1.tar.gz && mv aclImdb data/aclImdb/

# Make sure you have spacy english model downloaded
python -m spacy download en

wget -nH -r -np -P data/ http://files.fast.ai/models/wt103/