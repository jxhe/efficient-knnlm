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
dstore_weight="False"
moe_path="none"
probe=32
cache_feature="none"
cutoff=50
moe_feat="ctxt,lm_ent,lm_max,fert"
ckpt="knnlm_ckpt/wt103_checkpoint_best.pt"
dataset="wikitext-103"
gpu_index="False"

while getopts ":d:c:s:p:i:n:l:m:t:o:k:e:w:h:b:f:u:a:g:" arg; do
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
    w) dstore_weight="$OPTARG"
    ;;
    h) moe_path="$OPTARG"
    ;;
    b) probe="$OPTARG"
    ;;
    f) cache_feature="$OPTARG"
    ;;
    u) cutoff="$OPTARG"
    ;;
    a) moe_feat="$OPTARG"
    ;;
    g) gpu_index="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

if [ "${dataset}" = "law" ];
then
    max_tokens=2048
    ctxt=0
    mode="eos"
    # lmbda=0.9
    extra="--remove-bpe"
else
    extra=""
fi

python eval_lm.py data-bin/${dataset} \
    --path ${ckpt} \
    --sample-break-mode ${mode} --max-tokens ${max_tokens} \
    --context-window ${ctxt} --softmax-batch 1024 \
    --gen-subset ${split} --dstore-filename ${dstore_file} \
    --indexfile ${index_file} \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k ${k} --lmbda ${lmbda} --dstore-size ${dstore_size} --knn-keytype last_ffn_input \
    --probe ${probe} --knnlm --fp16 --dstore-fp16 --no-load-keys --knn-sim-func "do_not_recomp_l2" \
    --move-dstore-to-mem \
    --knn-temp ${temp} \
    --dstore-weight ${dstore_weight} \
    --ar-ckpt ${moe_path} \
    --ar-freq-dict ${cache_feature} \
    --ar-cutoff ${cutoff} \
    --ar-feat-type ${moe_feat} \
    --gpu-index ${gpu_index} \
    ${extra}
    # --dstore-weight \
