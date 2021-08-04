#!/bin/bash
DATE=`date +%Y%m%d`

# declare -a context_list=(2 4 10 100)
# arglen=${#context_list[@]}
# i=$(( taskid%arglen ))

seed=927
lr=0.0005
# feature="freq,lm_ent,lm_max"

declare -a hid_list=(128 256)
declare -a nl_list=(1 3 5)
declare -a bs_list=(64 128 1536)
# declare -a drop_list=(0.2)

dataset=$1
train=$2
val=$3
load=$4
l1=${5:-0}
feature=${6:-"all"}
ngram=${7:-0}

hid=128
nl=5
bs=64
drop=0.2
arch='mlp'

feature_str=$(printf "$feature" | tr , _)

# SAVE=checkpoint/${dataset}/${DATE}/moe/${arch}.l1.${l1}.hid${hid}.nl${nl}.bs${bs}.drop${drop}.ft${feature_str}.seed${seed}.jobid${SLURM_ARRAY_JOB_ID}.taskid${SLURM_ARRAY_TASK_ID}
# rm -r ${SAVE}; mkdir -p ${SAVE}

python knnlm_scripts/moe.py \
    --hidden-units ${hid} \
    --nlayers ${nl} \
    --dropout ${drop} \
    --seed ${seed} \
    --output-dir ${load} \
    --lr ${lr} \
    --feature-type ${feature} \
    --batch-size ${bs} \
    --arch ${arch} \
    --l1 ${l1} \
    --train ${train} \
    --val ${val} \
    --load-model ${load} \
    --ngram ${ngram} \
    --eval \
    # --val-others ${val_others} \
    # --train-others ${train_others} \
    # --debug
    # --input /projects/junxianh/knnlmXS/datasets/wikitext-103/test_feat_freq_normalized.jsonl,/projects/junxianh/knnlmXS/datasets/wikitext-103/valid_feat_freq_normalized.jsonl \
