#! /bin/bash
#
# eval_knnlm.sh
# Copyright (C) 2021-05-03 Junxian <He>
#
# Distributed under terms of the MIT license.
#




dataset=$1
ckpt=$2
split=${3:-"test"}
max_tokens=${4:-3072}
ctxt=${5:-0}

python eval_lm.py data-bin/${dataset} \
    --sample-break-mode eos \
    --path ${ckpt} \
    --max-tokens ${max_tokens} \
    --context-window ${ctxt} \
    --gen-subset ${split} \
    --softmax-batch 1024 \
    --remove-bpe \
    # --output-word-probs  \

