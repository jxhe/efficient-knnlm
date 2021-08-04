#!/bin/bash

# benchmark datastore pruning

dataset=wikitext-103
ckpt="knnlm_ckpt/wt103_checkpoint_best.pt"
split="valid"
k=1024
temp=1



dstore_size=xx
dstore_file=xx
index=xx

echo "evaluate knnlm with ${dstore_file}"

bash knnlm_scripts/utils_cmd/eval_knnlm.sh \
    -n ${dstore_size} \
    -p ${dstore_file} \
    -i ${index} \
    -d ${dataset} \
    -c ${ckpt} \
    -e ${temp} \
    -k ${k} \
    -s ${split} \
    -w "True" \

# done

