#!/bin/bash

echo "compute frequency/fertility dictionary"
python ef_knnlm/domain_adaptation/adaptive_retrieval/cache_freq_fertility.py \
    --data datasets/law/train.en.tokenized \
    --cache datasets/law \
    --dict-path data-bin/law/dict.txt \
    --csize 1 \
    --break-line \


cd datasets/law-valid
ln -s ../law/fertility_cache_id.pickle fertility_cache_id.pickle
ln -s ../law/freq_cache_id.pickle freq_cache_id.pickle
cd ../../

echo "save all features on the new retrieval adaptor training data"
bash ef_knnlm/domain_adaptation/adaptive_retrieval/precompute_all_features.sh train

echo "save all features on the new retrieval adaptor valid data"
bash ef_knnlm/domain_adaptation/adaptive_retrieval/precompute_all_features.sh valid
