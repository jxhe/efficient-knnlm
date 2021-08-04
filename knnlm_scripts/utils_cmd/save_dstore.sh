#!/bin/bash

# ctxt=${context_list[$i]}
# ctxt=10
# tokens_per_sample=$(( ctxt+1 ))

dataset=$1
dstore_size=$2
max_tokens=$3
ctxt=$4
tokens_per_sample=$5
output=$6
ckpt=$7
mode=${8:-"none"}

python eval_lm.py data-bin/${dataset} \
    --path ${ckpt} \
    --sample-break-mode ${mode} --max-tokens ${max_tokens} \
    --softmax-batch 1024 --gen-subset train \
    --context-window ${ctxt} --tokens-per-sample ${tokens_per_sample} \
    --dstore-mmap ${output} --knn-keytype 'last_ffn_input' \
    --dstore-size ${dstore_size}  \
    --log-interval 100 \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --fp16 --dstore-fp16 \
    --save-knnlm-dstore \

