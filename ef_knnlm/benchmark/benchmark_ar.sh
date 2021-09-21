#!/bin/bash

# benchmark adaptive retrieval


# the cutoff ratio in adaptive retrieval
# by default we cut off half of the retrieval
cutoff=50
ar_ckpt=$1

dstore_prefix=dstore/dstore_size103225485_embed1024_fp16
index_file=dstore/knn.103225485.pca512.m64.index


bash ef_knnlm/utils_cmd/eval_knnlm.sh \
    -d wikitext-103 \
    -s test \
    -p ${dstore_prefix} \
    -i ${index_file} \
    -c knnlm_ckpt/wt103_checkpoint_best.pt \
    -n 103225485 \
    -f datasets/wikitext-103 \
    -a ctxt,freq,lm_ent,lm_max,fert \
    -u ${cutoff} \
    -h ${ar_ckpt} \
