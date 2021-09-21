#!/bin/bash

split=$1

dataset=wikitext-103-valid
ckpt='knnlm_ckpt/wt103_checkpoint_best.pt'
index_file='dstore/knn.103225485.pca512.m64.index'
dstore_size=103225485
dstore_file='dstore/dstore_size103225485_embed1024_fp16'
max_tokens=3072
ctxt=2560
lmbda=0.25
mode="complete"

bash ef_knnlm/utils_cmd/precompute_all_features.sh \
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

