#!/bin/bash

split=$1
ckpt='knnlm_ckpt/wt103_checkpoint_best.pt'
dataset='wikitext103-valid'
index_file='dstore/knn.default.index'
dstore_file='dstore/dstore_size103225485_embed1024_fp16'
dstore_size=103225485
max_tokens=3072
ctxt=2560
lmbda=0.25
mode="complete"

bash knnlm_scripts/utils_cmd/precompute_all_features.sh \
    -d ${dataset} \
    -c ${ckpt} \
    -s ${split} \
    -p ${dstore_file} \
    -i ${index_file} \
    -n ${dstore_size} \
    -l ${lmbda} \
    -m ${max_tokens} \
    -t ${ctxt} \
    -o ${mode}

