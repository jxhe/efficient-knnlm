#!/bin/bash

echo "compute frequency/fertility dictionary"
python ef_knnlm/adaptive_retrieval/cache_freq_fertility.py \
    --data datasets/wikitext-103/wiki.train.tokens \
    --cache datasets/wikitext-103 \
    --dict-path data-bin/wikitext-103/dict.txt \
    --csize 1 


cd datasets/wikitext-103-valid
ln -s ../wikitext-103/fertility_cache_id.pickle fertility_cache_id.pickle
ln -s ../wikitext-103/freq_cache_id.pickle req_cache_id.pickle
cd ../../

echo "save all features on the new retrieval adaptor training data"
bash ef_knnlm/adaptive_retrieval/precompute_all_features.sh train

echo "save all features on the new retrieval adaptor valid data"
bash ef_knnlm/adaptive_retrieval/precompute_all_features.sh valid