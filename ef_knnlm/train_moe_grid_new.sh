#!/bin/bash


GPU=$1
seed=927
lr=0.0005
# feature="freq,lm_ent,lm_max"
feature=$2

DATE=`date +%Y%m%d`

declare -a hid_list=(128)
declare -a nl_list=(5)
# declare -a drop_list=(0.1 0.2 0.3 0.5)
declare -a activation_list=('relu')
declare -a drop_list=(0.2)

feature_str=$(printf "$feature" | tr , _)


for hid in "${hid_list[@]}"; do
    for nl in "${nl_list[@]}"; do
        for activation in "${activation_list[@]}"; do
            for drop in "${drop_list[@]}"; do
                SAVE=checkpoint/wikitext_103/${DATE}/moe/mlp.nh${hid}.nl${nl}.act${activation}.drop${drop}.ft${feature_str}.seed${seed}.gpu${GPU}
                rm -r ${SAVE}; mkdir -p ${SAVE}

                CUDA_VISIBLE_DEVICES=${GPU} python knnlm_scripts/moe.py \
                    --hidden-units ${hid} \
                    --nlayers ${nl} \
                    --dropout ${drop} \
                    --seed ${seed} \
                    --output-dir ${SAVE} \
                    --activation ${activation} \
                    --lr ${lr} \
                    --feature-type ${feature} \
                    --train  /projects/junxianh/knnlmXS/datasets/wikitext-103/valid_feat \
                    --val /projects/junxianh/knnlmXS/datasets/wikitext-103/test_feat \
                    # --debug
                    # --input /projects/junxianh/knnlmXS/datasets/wikitext-103/test_feat_freq_normalized.jsonl,/projects/junxianh/knnlmXS/datasets/wikitext-103/valid_feat_freq_normalized.jsonl \
            done
        done
    done
done
