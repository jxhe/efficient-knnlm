#! /bin/bash
#
# eval_knnlm.sh
# Copyright (C) 2021-05-03 Junxian <He>
#
# Distributed under terms of the MIT license.
#



temp=1
k=1024
max_tokens=3072
ctxt=2560
mode="complete"
split="test"
lmbda=0.25

while getopts ":d:c:s:p:i:n:l:m:t:o:k:e:" arg; do
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
    k) k="$OPTARG"
    ;;
    e) temp="$OPTARG"
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
    --k ${k} --lmbda ${lmbda} --dstore-size ${dstore_size} --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 \
    --knn-temp ${temp} \
    # --dstore-weight \
