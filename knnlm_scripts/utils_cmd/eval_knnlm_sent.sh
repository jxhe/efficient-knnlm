#! /bin/bash
#
# eval_knnlm.sh
# Copyright (C) 2021-05-03 Junxian <He>
#
# Distributed under terms of the MIT license.
#



dstore_size=$1
dstore_file=$2
index_file=$3
dataset=$4
ckpt=$5
temp=${6:-1}
split=${7:-"test"}
lmbda=${8:-0.25}

python eval_lm.py data-bin/${dataset} \
    --path ${ckpt} \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset ${split} --dstore-filename ${dstore_file} \
    --indexfile ${index_file} \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda ${lmbda} --dstore-size ${dstore_size} --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --no-load-keys --knn-sim-func "do_not_recomp_l2" \
    --knn-temp ${temp} \
    # --dstore-weight \
