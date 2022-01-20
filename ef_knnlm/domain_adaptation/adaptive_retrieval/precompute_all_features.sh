#!/bin/bash

split=$1

dataset=law-valid
ckpt='/projects/tir4/users/junxianh/projects/knnlmXS/checkpoint/law/20210509/checkpoint_best.pt'
index_file='/projects/tir4/users/urialon/efficient-knnlm/checkpoints/law/knn_finetuned'
dstore_size=19068709
dstore_file='/projects/tir4/users/urialon/efficient-knnlm/checkpoints/law/dstore16_finetuned_size19068709_embed1536_fp16'
max_tokens=2048
ctxt=0
lmbda=0.25
mode="eos"

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

