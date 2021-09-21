#!/bin/bash

# dataset=$1
# ckpt=$2
# split=$3
# dstore_file=${4}
# index_file=${5}
# dstore_size=${6}
lmbda=0.25
max_tokens=3072
ctxt=2560
mode="complete"

while getopts ":d:c:s:p:i:n:l:m:t:o:" arg; do
  case $arg in
    d) dataset="$OPTARG"
    ;;
    c) ckpt="$OPTARG"
    ;;
    s) split="$OPTARG"
    ;;
    p) dstore_file="$OPTARG"
    ;;
    i) index_file="$OPTARG"
    ;;
    n) dstore_size="$OPTARG"
    ;;
    l) lmbda="$OPTARG"
    ;;
    m) max_tokens="$OPTARG"
    ;;
    t) ctxt="$OPTARG"
    ;;
    o) mode="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

python eval_lm.py data-bin/${dataset} \
    --path ${ckpt} \
    --sample-break-mode ${mode} --max-tokens ${max_tokens} \
    --context-window ${ctxt} --softmax-batch 1024 \
    --gen-subset ${split} --dstore-filename ${dstore_file} \
    --indexfile ${index_file} \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda ${lmbda} --dstore-size ${dstore_size} --knn-keytype last_ffn_input \
    --probe 32 --fp16 --dstore-fp16 --no-load-keys --knn-sim-func "do_not_recomp_l2" \
    --save-feature datasets/${dataset}/${split} --ar-feat-cache datasets/${dataset} \
    --knnlm \
