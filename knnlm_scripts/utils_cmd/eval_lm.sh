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
ctxt=${5:-2560}
mode=${6:-"complete"}

if [ "${dataset}" = "law" ];
then
    max_tokens=2048
    ctxt=0
    mode="eos"
    extra="--remove-bpe"
else
    extra=""
fi

python eval_lm.py data-bin/${dataset} \
    --sample-break-mode ${mode} \
    --path ${ckpt} \
    --max-tokens ${max_tokens} \
    --context-window ${ctxt} \
    --gen-subset ${split} \
    --softmax-batch 1024 \
    --fp16 \
    ${extra} \
    # --remove-bpe \
    # --output-word-probs  \

